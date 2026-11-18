use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bytemuck::cast_slice;
use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::journal::{Frame, Journal};
use crate::meta::{Meta, FORMAT, MAX_DIM};
use crate::quant::{self, QUANT_VERSION};
use crate::scan;

pub const F32_FILE: &str = "vectors.f32";
fn slab_backup(dir: &Path, slab: u32) -> std::path::PathBuf {
    dir.join(format!("vectors.f32.s{slab}.old"))
}
const SQ8_FILE: &str = "vectors.i8";
const SQ8_TMP: &str = "vectors.i8.tmp";
const JOURNAL_FILE: &str = "journal.bin";
const LOCK_FILE: &str = "lock";
const MAX_SLABS: usize = 64;
const SQ8_MIN_ALIVE: u32 = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repr {
    F32,
    Sq8,
}

pub struct Config {
    pub dim: u32,
    pub model_hash: String,
    pub initial_capacity: u32,
    pub slab_rows: Option<u32>,
}

impl Config {
    pub fn new(dim: u32, model_hash: impl Into<String>) -> Self {
        Config {
            dim,
            model_hash: model_hash.into(),
            initial_capacity: 1024,
            slab_rows: None,
        }
    }

    pub fn with_slab_rows(mut self, rows: u32) -> Self {
        self.slab_rows = Some(rows.max(1));
        self
    }

    pub fn with_capacity(mut self, cap: u32) -> Self {
        self.initial_capacity = cap;
        self
    }
}

pub struct OpenOptions {
    pub expected_model_hash: String,
}

impl OpenOptions {
    pub fn new(expected_model_hash: impl Into<String>) -> Self {
        OpenOptions {
            expected_model_hash: expected_model_hash.into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JobProgress {
    pub done: u32,
    pub total: u32,
}

#[derive(Default)]
pub struct JobOptions<'a> {
    pub progress: Option<&'a dyn Fn(JobProgress)>,
    pub cancel: Option<&'a AtomicBool>,
}

impl JobOptions<'_> {
    fn report(&self, done: u32, total: u32) {
        if let Some(f) = self.progress {
            f(JobProgress { done, total });
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.map_or(false, |c| c.load(Ordering::Relaxed))
    }
}

#[derive(Default)]
pub struct SearchOptions {
    pub repr: Option<Repr>,
    pub rerank: Option<usize>,
    pub threads: Option<usize>,
}

pub struct SearchResult {
    pub slots: Vec<u32>,
    pub scores: Vec<f32>,
    pub repr: Repr,
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub dim: u32,
    pub n: u32,
    pub alive: u32,
    pub capacity: u32,
    pub repr: Repr,
}

pub struct Store {
    dir: PathBuf,
    _lock: File,
    inner: RwLock<Inner>,
    slabs: Box<[RwLock<()>]>,
    slab_rows: u32,
}

struct Quant {
    stride: usize,
    file: File,
    map: Arc<Mmap>,
}

fn write_at(file: &File, buf: &[u8], off: u64) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(file, &buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(file, &buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "写 0 字节",
            ));
        }
        done += n;
    }
    Ok(())
}

struct Inner {
    meta: Meta,
    stride: usize,
    file: File,
    map: Option<Arc<Mmap>>,
    quant: Option<Quant>,
    journal: Journal,
}

impl Inner {
    fn dim(&self) -> usize {
        self.meta.dim as usize
    }

    fn n(&self) -> u32 {
        self.meta.watermark()
    }

    fn capacity(&self) -> u32 {
        self.meta.capacity()
    }

    fn slab_count(&self) -> u32 {
        self.meta.slabs.len() as u32
    }

    fn rows_allocated(&self) -> u32 {
        self.meta.slabs.iter().sum()
    }

    fn slab_range(&self, slab: u32) -> (u32, u32) {
        self.meta.slab_range(slab)
    }

    fn repr(&self) -> Repr {
        if self.quant.is_some() {
            Repr::Sq8
        } else {
            Repr::F32
        }
    }

    fn map(&self) -> &Mmap {
        self.map.as_ref().expect("store is mapped")
    }

    fn rows_all(&self) -> &[f32] {
        let len = self.capacity() as usize * self.stride;
        cast_slice(&self.map()[..len])
    }

    fn row_at(&self, slot: u32) -> &[f32] {
        let d = self.dim();
        let s = slot as usize * d;
        &self.rows_all()[s..s + d]
    }

    fn is_dead(&self, slot: u32) -> bool {
        self.row_at(slot)[0] == 0.0
    }

    fn commit(&self) -> Result<()> {
        self.file.sync_all()?;
        if let Some(q) = &self.quant {
            q.file.sync_all()?;
        }
        Ok(())
    }

    fn alive(&self) -> u32 {
        self.rows_allocated().saturating_sub(self.meta.deleted)
    }

    fn recount_alive(&self) -> u32 {
        let dim = self.dim();
        let mut dead = 0u32;
        for slab in 0..self.slab_count() {
            let (lo, hi) = self.slab_range(slab);
            dead += self.rows_all()[lo as usize * dim..hi as usize * dim]
                .chunks_exact(dim)
                .filter(|r| r[0] == 0.0)
                .count() as u32;
        }
        dead
    }

    fn suggest(&self, alive: u32) -> Repr {
        if self.quant.is_some() && alive >= SQ8_MIN_ALIVE {
            Repr::Sq8
        } else {
            Repr::F32
        }
    }

    fn write_f32_rows(&self, start: u32, data: &[f32]) -> Result<()> {
        let off = start as usize * self.stride;
        write_at(&self.file, cast_slice::<f32, u8>(data), off as u64)?;
        if let Some(q) = &self.quant {
            let dim = self.meta.dim as usize;
            let mut buf = vec![0u8; q.stride];
            for (i, row) in data.chunks_exact(dim).enumerate() {
                quant::encode(row, &mut buf);
                let off = (start as usize + i) * q.stride;
                write_at(&q.file, &buf, off as u64)?;
            }
        }
        Ok(())
    }

    fn kill_row(&self, slot: u32) -> Result<()> {
        let zeros = vec![0u8; self.stride];
        write_at(&self.file, &zeros, (slot as usize * self.stride) as u64)?;
        if let Some(q) = &self.quant {
            let zeros = vec![0u8; q.stride];
            write_at(&q.file, &zeros, (slot as usize * q.stride) as u64)?;
        }
        Ok(())
    }
}

fn open_lock(dir: &Path) -> Result<File> {
    let f = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(dir.join(LOCK_FILE))?;
    match f.try_lock() {
        Ok(()) => Ok(f),
        Err(fs::TryLockError::WouldBlock) => Err(Error::Locked),
        Err(fs::TryLockError::Error(e)) => Err(Error::Io(e)),
    }
}

fn invalid(msg: &'static str) -> Error {
    Error::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 8))
        .unwrap_or(1)
}

fn grow(inner: &mut Inner, need: u32) -> Result<()> {
    let sr = inner.meta.slab_rows;
    while inner.capacity() < need {
        inner.meta.slabs.push(0);
    }
    let cap = inner.capacity();
    debug_assert!(sr > 0);
    inner.file.set_len(cap as u64 * inner.stride as u64)?;
    inner.map = Some(Arc::new(unsafe { Mmap::map(&inner.file)? }));
    if let Some(q) = &mut inner.quant {
        q.file.set_len(cap as u64 * q.stride as u64)?;
        q.map = Arc::new(unsafe { Mmap::map(&q.file)? });
    }
    Ok(())
}

fn normalize(vecs: &[f32], dim: usize) -> Result<Vec<f32>> {
    if dim == 0 || vecs.len() % dim != 0 {
        return Err(Error::DimMismatch);
    }
    let mut out = vec![0f32; vecs.len()];
    for (i, row) in vecs.chunks_exact(dim).enumerate() {
        if !row.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let mut sq = 0f64;
        for &v in row {
            sq += (v as f64) * (v as f64);
        }
        if !(sq >= 1e-24) {
            return Err(Error::DegenerateVector);
        }
        let inv = (1.0 / sq.sqrt()) as f32;
        let dst = &mut out[i * dim..(i + 1) * dim];
        for (d, &v) in dst.iter_mut().zip(row) {
            *d = v * inv;
        }
        if dst[0] == 0.0 {
            dst[0] = 1e-30;
        }
    }
    Ok(out)
}

fn recover_compact(dir: &Path, meta: &mut Meta) -> Result<()> {
    let stride = meta.dim as u64 * 4;
    if let Some(slab) = meta.compacting {
        let bak = slab_backup(dir, slab);
        if bak.exists() {
            let bytes = fs::read(&bak)?;
            let (lo, _) = meta.slab_range(slab);
            let off = lo as u64 * stride;
            let f = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(dir.join(F32_FILE))?;
            write_at(&f, &bytes, off)?;
            f.sync_all()?;
        }
        if meta.quant_version.is_some() {
            meta.quant_version = None;
        }
        meta.compacting = None;
        meta.write(dir)?;
    }
    for e in fs::read_dir(dir)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("vectors.f32.s") && name.ends_with(".old") {
            let _ = fs::remove_file(e.path());
        }
    }
    Ok(())
}

fn open_quant(dir: &Path, capacity: u32, dim: usize) -> Result<Option<Quant>> {
    let path = dir.join(SQ8_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let stride = quant::stride(dim);
    let file = fs::OpenOptions::new().read(true).write(true).open(&path)?;
    let len = file.metadata()?.len();
    if len % stride as u64 != 0 || (len / stride as u64) > u32::MAX as u64 {
        fs::remove_file(&path)?;
        return Ok(None);
    }
    if (len / stride as u64) as u32 != capacity {
        file.set_len(capacity as u64 * stride as u64)?;
    }
    let map = Arc::new(unsafe { Mmap::map(&file)? });
    Ok(Some(Quant { stride, file, map }))
}


const JOB_CHUNK: u32 = 4096;

fn build_quant(inner: &mut Inner, dir: &Path, opts: &JobOptions) -> Result<()> {
    let dim = inner.dim();
    let qstride = quant::stride(dim);
    let cap = inner.capacity() as usize;
    let total: u32 = inner.meta.slabs.iter().sum();

    let tmp = dir.join(SQ8_TMP);
    {
        let f = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        f.set_len((cap * qstride) as u64)?;
        let mut buf = vec![0u8; JOB_CHUNK as usize * qstride];
        let mut done = 0u32;
        opts.report(0, total);
        for slab in 0..inner.slab_count() {
            let (lo, hi) = inner.slab_range(slab);
            let mut at = lo;
            while at < hi {
                if opts.cancelled() {
                    drop(f);
                    let _ = fs::remove_file(&tmp);
                    return Err(Error::Cancelled);
                }
                let n = (hi - at).min(JOB_CHUNK);
                let rows = inner.rows_all();
                for i in 0..n as usize {
                    let d = (at as usize + i) * dim;
                    quant::encode(&rows[d..d + dim], &mut buf[i * qstride..(i + 1) * qstride]);
                }
                write_at(&f, &buf[..n as usize * qstride], at as u64 * qstride as u64)?;
                at += n;
                done += n;
                opts.report(done, total);
            }
        }
        f.sync_all()?;
    }

    inner.quant = None;
    let path = dir.join(SQ8_FILE);
    fs::rename(&tmp, &path)?;
    let file = fs::OpenOptions::new().read(true).write(true).open(&path)?;
    let map = Arc::new(unsafe { Mmap::map(&file)? });
    inner.quant = Some(Quant {
        stride: qstride,
        file,
        map,
    });
    inner.meta.quant_version = Some(QUANT_VERSION);
    inner.meta.write(dir)
}

impl Store {
    pub fn create(path: impl AsRef<Path>, cfg: Config) -> Result<Store> {
        let dir = path.as_ref().to_path_buf();
        if cfg.dim == 0 || cfg.dim > MAX_DIM {
            return Err(Error::InvalidConfig);
        }
        fs::create_dir_all(&dir)?;
        if dir.join(crate::meta::META_FILE).exists() {
            return Err(Error::InvalidConfig);
        }
        let lock = open_lock(&dir)?;
        let stride = cfg.dim as usize * 4;
        let slab_rows = cfg
            .slab_rows
            .unwrap_or_else(|| (cfg.initial_capacity / 16).clamp(4096, 1 << 20));
        let n_slabs = cfg.initial_capacity.max(1).div_ceil(slab_rows).max(1);
        let meta = Meta {
            format: FORMAT.to_string(),
            dim: cfg.dim,
            slab_rows,
            slabs: vec![0; n_slabs as usize],
            model_hash: cfg.model_hash,
            quant_version: None,
            deleted: 0,
            compacting: None,
        };
        let capacity = meta.capacity();
        let file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(dir.join(F32_FILE))?;
        file.set_len(capacity as u64 * stride as u64)?;
        file.sync_all()?;
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let journal = Journal::open(dir.join(JOURNAL_FILE))?;
        meta.write(&dir)?;
        Ok(Store {
            dir,
            _lock: lock,
            slabs: (0..MAX_SLABS).map(|_| RwLock::new(())).collect(),
            slab_rows,
            inner: RwLock::new(Inner {
                meta,
                stride,
                file,
                map: Some(map),
                quant: None,
                journal,
            }),
        })
    }

    pub fn open(path: impl AsRef<Path>, opts: OpenOptions) -> Result<Store> {
        let dir = path.as_ref().to_path_buf();
        let mut meta = Meta::read(&dir)?;
        if meta.model_hash != opts.expected_model_hash {
            return Err(Error::ModelMismatch);
        }
        let lock = open_lock(&dir)?;
        recover_compact(&dir, &mut meta)?;
        for t in [SQ8_TMP] {
            if dir.join(t).exists() {
                fs::remove_file(dir.join(t))?;
            }
        }

        let stride = meta.dim as usize * 4;
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(F32_FILE))?;
        let len = file.metadata()?.len();
        if len % stride as u64 != 0 {
            return Err(invalid("vectors.f32 长度不是 stride 的整数倍"));
        }
        let capacity = (len / stride as u64) as u32;
        if capacity < meta.capacity() {
            file.set_len(meta.capacity() as u64 * stride as u64)?;
        } else if capacity > meta.capacity() {
            file.set_len(meta.capacity() as u64 * stride as u64)?;
        }
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let journal = Journal::open(dir.join(JOURNAL_FILE))?;
        let mut inner = Inner {
            meta,
            stride,
            file,
            map: Some(map),
            quant: None,
            journal,
        };

        let has_quant = dir.join(SQ8_FILE).exists();
        match has_quant {
            true if inner.meta.quant_version != Some(QUANT_VERSION) => {
                fs::remove_file(dir.join(SQ8_FILE))?;
                if inner.meta.quant_version.take().is_some() {
                    inner.meta.write(&dir)?;
                }
            }
            true => {
                inner.quant = open_quant(&dir, capacity, inner.dim())?;
                if inner.quant.is_none() {
                    inner.meta.quant_version = None;
                    inner.meta.write(&dir)?;
                }
            }
            false => {
                if inner.meta.quant_version.take().is_some() {
                    inner.meta.write(&dir)?;
                }
            }
        }

        if let Some(frames) = inner.journal.read_batch()? {
            for f in &frames {
                apply_frame(&inner, f)?;
            }
            inner.commit()?;
            inner.journal.truncate()?;
            let dead = inner.recount_alive();
            if dead != inner.meta.deleted {
                inner.meta.deleted = dead;
                inner.meta.write(&dir)?;
            }
        } else {
            inner.journal.truncate()?;
        }
        let slab_rows = inner.meta.slab_rows;

        Ok(Store {
            dir,
            _lock: lock,
            slabs: (0..MAX_SLABS).map(|_| RwLock::new(())).collect(),
            slab_rows,
            inner: RwLock::new(inner),
        })
    }

    pub fn append(&self, vecs: &[f32]) -> Result<u32> {
        let dim = self.dim() as usize;
        let normalized = normalize(vecs, dim)?;
        let m = (normalized.len() / dim) as u32;

        loop {
            let start = self.inner.read().unwrap().n();
            let first = self.slab_of(start);
            let last = if m == 0 {
                first
            } else {
                self.slab_of(start + m - 1)
            };
            let _slabs = self.write_slabs(first, last);

            let mut inner = self.inner.write().unwrap();
            if inner.n() != start {
                continue;
            }
            let need = start.checked_add(m).ok_or(Error::InvalidConfig)?;
            if need > inner.capacity() {
                grow(&mut inner, need)?;
            }
            inner.write_f32_rows(start, &normalized)?;
            inner.commit()?;
            let sr = inner.meta.slab_rows;
            let last = (need - 1) / sr;
            for i in 0..=last {
                let lo = i * sr;
                inner.meta.slabs[i as usize] = ((i + 1) * sr).min(need) - lo;
            }
            inner.meta.write(&self.dir)?;

            return Ok(start);
        }
    }

    pub fn overwrite(&self, slots: &[u32], vecs: &[f32]) -> Result<()> {
        let dim = self.dim() as usize;
        if vecs.len() != slots.len() * dim {
            return Err(Error::DimMismatch);
        }
        let normalized = normalize(vecs, dim)?;
        let (lo, hi) = self.span_of(slots);
        let _slabs = self.write_slabs(lo, hi);

        let mut inner = self.inner.write().unwrap();
        for &s in slots {
            if s >= inner.n() || inner.is_dead(s) {
                return Err(Error::InvalidSlot);
            }
        }
        let frames: Vec<Frame> = slots
            .iter()
            .zip(normalized.chunks_exact(dim))
            .map(|(&slot, row)| Frame::Overwrite {
                slot,
                row: row.to_vec(),
            })
            .collect();
        inner.journal.write_batch(&frames)?;
        inner.journal.commit()?;
        for f in &frames {
            apply_frame(&inner, f)?;
        }
        inner.commit()?;
        inner.journal.truncate()?;
        Ok(())
    }

    pub fn delete(&self, slots: &[u32]) -> Result<()> {
        let (lo, hi) = self.span_of(slots);
        let _slabs = self.write_slabs(lo, hi);
        let mut inner = self.inner.write().unwrap();
        let n = inner.n();
        let mut dead = Vec::new();
        for &s in slots {
            if s >= n {
                return Err(Error::InvalidSlot);
            }
            if !inner.is_dead(s) {
                dead.push(s);
            }
        }
        if dead.is_empty() {
            return Ok(());
        }
        let frames: Vec<Frame> = dead.iter().map(|&slot| Frame::Delete { slot }).collect();
        inner.journal.write_batch(&frames)?;
        inner.journal.commit()?;
        for f in &frames {
            apply_frame(&inner, f)?;
        }
        inner.commit()?;
        inner.meta.deleted += dead.len() as u32;
        inner.meta.write(&self.dir)?;
        inner.journal.truncate()?;
        Ok(())
    }

    pub fn compact_slab(&self, slab: u32) -> Result<(Vec<u32>, Vec<u32>)> {
        self.compact_slab_with(slab, &JobOptions::default())
    }

    pub fn compact_slab_with(
        &self,
        slab: u32,
        opts: &JobOptions,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let _slabs = self.write_slabs(slab, slab);
        let mut inner = self.inner.write().unwrap();
        if slab >= inner.slab_count() {
            return Err(Error::InvalidSlot);
        }
        let (lo, hi) = inner.slab_range(slab);
        let dim = inner.dim();
        let stride = inner.stride;
        let alive: Vec<u32> = (lo..hi).filter(|&s| !inner.is_dead(s)).collect();
        if alive.len() as u32 == hi - lo {
            return Ok((Vec::new(), Vec::new()));
        }

        let bak = slab_backup(&self.dir, slab);
        {
            let mut f = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&bak)?;
            let base: &[f32] = inner.rows_all();
            let d0 = lo as usize * inner.dim();
            let d1 = hi as usize * inner.dim();
            f.write_all(cast_slice::<f32, u8>(&base[d0..d1]))?;
            f.sync_all()?;
        }
        inner.meta.compacting = Some(slab);
        inner.meta.write(&self.dir)?;
        let mut wrote = 0u32;
        opts.report(0, alive.len() as u32);
        while wrote < alive.len() as u32 {
            if opts.cancelled() {
                abort_slab(&mut inner, &self.dir, slab, &bak)?;
                return Err(Error::Cancelled);
            }
            let n = (alive.len() as u32 - wrote).min(JOB_CHUNK);
            let (fbuf, qbuf) = {
                let rows = inner.rows_all();
                let mut fbuf = vec![0u8; n as usize * stride];
                let mut qbuf = inner
                    .quant
                    .as_ref()
                    .map(|q| vec![0u8; n as usize * q.stride]);
                for i in 0..n as usize {
                    let s = alive[wrote as usize + i];
                    let d = s as usize * dim;
                    fbuf[i * stride..(i + 1) * stride]
                        .copy_from_slice(cast_slice::<f32, u8>(&rows[d..d + dim]));
                    if let (Some(q), Some(qb)) = (&inner.quant, qbuf.as_mut()) {
                        quant::encode(&rows[d..d + dim], &mut qb[i * q.stride..]);
                    }
                }
                (fbuf, qbuf)
            };
            let at = lo + wrote;
            write_at(&inner.file, &fbuf, at as u64 * stride as u64)?;
            if let (Some(q), Some(qbuf)) = (&inner.quant, qbuf.as_ref()) {
                write_at(&q.file, qbuf, at as u64 * q.stride as u64)?;
            }
            wrote += n;
            opts.report(wrote, alive.len() as u32);
        }
        let tail = (hi - lo - alive.len() as u32) as usize;
        if tail > 0 {
            write_at(
                &inner.file,
                &vec![0u8; tail * stride],
                (hi as u64 - tail as u64) * stride as u64,
            )?;
            if let Some(q) = &inner.quant {
                write_at(
                    &q.file,
                    &vec![0u8; tail * q.stride],
                    (hi as u64 - tail as u64) * q.stride as u64,
                )?;
            }
        }
        inner.commit()?;
        let dropped = (hi - lo) - alive.len() as u32;
        inner.meta.slabs[slab as usize] = alive.len() as u32;
        inner.meta.deleted = inner.meta.deleted.saturating_sub(dropped);
        inner.meta.compacting = None;
        inner.meta.write(&self.dir)?;
        let _ = fs::remove_file(&bak);

        Ok((alive, (lo..lo + inner.meta.slabs[slab as usize]).collect()))
    }

    pub fn compact(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        let slabs = self.inner.read().unwrap().slab_count();
        let (mut old, mut new) = (Vec::new(), Vec::new());
        for slab in 0..slabs {
            let (o, n) = self.compact_slab(slab)?;
            old.extend(o);
            new.extend(n);
        }
        Ok((old, new))
    }

    pub fn slab_info(&self) -> (u32, u32) {
        let g = self.inner.read().unwrap();
        (g.slab_count(), g.meta.slab_rows)
    }

    fn slab_of(&self, slot: u32) -> u32 {
        (slot / self.slab_rows).min(MAX_SLABS as u32 - 1)
    }

    fn read_slab(&self, slab: u32) -> RwLockReadGuard<'_, ()> {
        self.slabs[slab as usize].read().unwrap()
    }

    fn write_slabs(&self, first: u32, last: u32) -> Vec<RwLockWriteGuard<'_, ()>> {
        (first..=last)
            .map(|i| self.slabs[i as usize].write().unwrap())
            .collect()
    }

    fn span_of(&self, slots: &[u32]) -> (u32, u32) {
        if slots.is_empty() {
            return (0, 0);
        }
        let mut lo = u32::MAX;
        let mut hi = 0u32;
        for &s in slots {
            lo = lo.min(self.slab_of(s));
            hi = hi.max(self.slab_of(s));
        }
        (lo, hi)
    }

    fn all_read_slabs(&self, last: u32) -> Vec<RwLockReadGuard<'_, ()>> {
        (0..=last).map(|i| self.read_slab(i)).collect()
    }

    pub fn search(&self, q: &[f32], k: usize, opts: SearchOptions) -> Result<SearchResult> {
        let (dim, n, stride, f32_map, quant) = {
            let g = self.inner.read().unwrap();
            (
                g.dim(),
                g.n(),
                g.stride,
                g.map.clone().expect("mapped"),
                g.quant.as_ref().map(|q| (q.stride, q.map.clone())),
            )
        };
        if q.len() != dim {
            return Err(Error::DimMismatch);
        }
        if n == 0 {
            return Err(Error::Empty);
        }
        let qn = normalize(q, dim)?;
        let alive = self.inner.read().unwrap().alive();

        let actual = if quant.is_some() { Repr::Sq8 } else { Repr::F32 };
        let repr = match opts.repr {
            Some(Repr::F32) => Repr::F32,
            Some(r) if r != actual => actual, // 未物化 → 退回实际可用的
            Some(r) => r,
            None => self.inner.read().unwrap().suggest(alive),
        };
        let depth = opts
            .rerank
            .or(match repr {
                Repr::Sq8 => Some((2 * k).max(1000)),
                Repr::F32 => None,
            });
        let kk = k.max(depth.unwrap_or(0)).min(n as usize);
        if kk == 0 {
            return Ok(SearchResult {
                slots: Vec::new(),
                scores: Vec::new(),
                repr,
            });
        }
        let threads = opts.threads.unwrap_or_else(default_threads);
        let last_slab = self.slab_of(n - 1);

        let (qi8, inv_q) = match repr {
            Repr::Sq8 => {
                let (a, b) = quant::encode_query(&qn)?;
                (Some(a), b)
            }
            Repr::F32 => (None, 0.0),
        };

        let mut hits: Vec<scan::Hit> = Vec::new();
        let mut dead = 0u32;
        for slab in 0..=last_slab {
            let lo = slab * self.slab_rows;
            let hi = ((slab + 1) * self.slab_rows).min(n);
            let rows = (hi - lo) as usize;
            let _g = self.read_slab(slab);
            let (h, d) = match repr {
                Repr::F32 => scan::top_k(
                    cast_slice(&f32_map[lo as usize * stride..hi as usize * stride]),
                    dim,
                    rows,
                    &qn,
                    kk,
                    threads,
                    lo,
                ),
                Repr::Sq8 => {
                    let (qstride, qmap) = quant.as_ref().expect("quant");
                    scan::top_k_i8(
                        &qmap[lo as usize * qstride..hi as usize * qstride],
                        *qstride,
                        dim,
                        rows,
                        qi8.as_ref().expect("q"),
                        inv_q,
                        kk,
                        threads,
                        lo,
                    )
                }
            };
            hits.extend(h);
            dead += d;
        }
        let _ = dead;

        hits.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.slot.cmp(&b.slot))
        });
        hits.truncate(kk);

        let do_rerank = repr != Repr::F32 && depth.map_or(false, |d| d > 0);
        let mut ranked: Vec<(f32, u32)> = if do_rerank && !hits.is_empty() {
            let _guards = self.all_read_slabs(last_slab);
            let base: &[f32] = cast_slice(&f32_map[..n as usize * stride]);
            let mut slots: Vec<u32> = hits.iter().map(|h| h.slot).collect();
            slots.sort_unstable();
            let mut scored: Vec<(f32, u32)> = slots
                .into_iter()
                .map(|s| {
                    let lo = s as usize * dim;
                    (scan::dot(&base[lo..lo + dim], &qn), s)
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            scored
        } else {
            hits.into_iter().map(|h| (h.score, h.slot)).collect()
        };
        ranked.truncate(k);

        let mut slots = Vec::with_capacity(ranked.len());
        let mut scores = Vec::with_capacity(ranked.len());
        for (score, slot) in ranked {
            slots.push(slot);
            scores.push(score);
        }
        Ok(SearchResult {
            slots,
            scores,
            repr,
        })
    }

    pub fn vectors(&self, slots: &[u32]) -> Result<Vec<f32>> {
        let inner = self.inner.read().unwrap();
        let dim = inner.dim();
        let mut out = Vec::with_capacity(slots.len() * dim);
        for &s in slots {
            if s >= inner.n() || inner.is_dead(s) {
                return Err(Error::InvalidSlot);
            }
            out.extend_from_slice(inner.row_at(s));
        }
        Ok(out)
    }

    pub fn warm(&self) -> Result<u32> {
        let mut inner = self.inner.write().unwrap();
        let dead = inner.recount_alive();
        if dead != inner.meta.deleted {
            inner.meta.deleted = dead;
            inner.meta.write(&self.dir)?;
        }
        Ok(inner.alive())
    }

    pub fn suggest_mode(&self) -> Repr {
        let inner = self.inner.read().unwrap();
        inner.suggest(inner.alive())
    }

    pub fn convert(&self, target: Repr) -> Result<()> {
        self.convert_with(target, &JobOptions::default())
    }

    pub fn convert_with(&self, target: Repr, opts: &JobOptions) -> Result<()> {
        let _slabs = self.write_slabs(0, MAX_SLABS as u32 - 1);
        let mut inner = self.inner.write().unwrap();
        match target {
            Repr::Sq8 => build_quant(&mut inner, &self.dir, opts),
            Repr::F32 => {
                if inner.quant.is_some() {
                    inner.meta.quant_version = None;
                    inner.meta.write(&self.dir)?; // 提交点：先改 meta
                    inner.quant = None;
                    let _ = fs::remove_file(self.dir.join(SQ8_FILE));
                }
                Ok(())
            }
        }
    }

    pub fn stats(&self) -> Stats {
        let inner = self.inner.read().unwrap();
        Stats {
            dim: inner.meta.dim,
            n: inner.n(),
            alive: inner.alive(),
            capacity: inner.capacity(),
            repr: inner.repr(),
        }
    }

    pub fn model_hash(&self) -> String {
        self.inner.read().unwrap().meta.model_hash.clone()
    }

    pub fn dim(&self) -> u32 {
        self.inner.read().unwrap().meta.dim
    }

    pub fn n(&self) -> u32 {
        self.inner.read().unwrap().n()
    }
}

fn abort_slab(inner: &mut Inner, dir: &Path, slab: u32, bak: &Path) -> Result<()> {
    let stride = inner.stride;
    let (lo, _) = inner.slab_range(slab);
    if bak.exists() {
        let bytes = fs::read(bak)?;
        write_at(&inner.file, &bytes, lo as u64 * stride as u64)?;
        inner.file.sync_all()?;
        let _ = fs::remove_file(bak);
    }
    if let Some(q) = inner.quant.take() {
        let path = dir.join(SQ8_FILE);
        drop(q);
        let _ = fs::remove_file(path);
        inner.meta.quant_version = None;
    }
    inner.meta.compacting = None;
    inner.meta.write(dir)?;
    Ok(())
}

fn apply_frame(inner: &Inner, f: &Frame) -> Result<()> {
    match f {
        Frame::Overwrite { slot, row } => {
            if *slot >= inner.n() || row.len() != inner.dim() {
                return Err(Error::InvalidSlot);
            }
            inner.write_f32_rows(*slot, row)?;
        }
        Frame::Delete { slot } => {
            if *slot >= inner.n() {
                return Err(Error::InvalidSlot);
            }
            inner.kill_row(*slot)?;
        }
    }
    Ok(())
}
