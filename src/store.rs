use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bytemuck::cast_slice;
use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::journal::{Frame, Journal};
use crate::meta::{Meta, FORMAT, MAX_DIM};
use crate::quant::{self, QUANT_VERSION};
use crate::scan;

pub const F32_FILE: &str = "vectors.f32";
/// 片压实时的原片备份：`vectors.f32.s<N>.old`
fn slab_backup(dir: &Path, slab: u32) -> std::path::PathBuf {
    dir.join(format!("vectors.f32.s{slab}.old"))
}
const SQ8_FILE: &str = "vectors.i8";
const SQ8_TMP: &str = "vectors.i8.tmp";
const JOURNAL_FILE: &str = "journal.bin";
const LOCK_FILE: &str = "lock";
const ALIVE_UNKNOWN: u32 = u32::MAX;
/// 片级锁的槽数上限（超出部分共用最后一把锁，只是锁变粗）。
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
    /// 片大小（行）。决定槽位 → 行的映射，写入 meta；缺省按 initial_capacity/16 取。
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

/// 长任务（convert / compact）的进度与取消。
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
    /// 元数据 / 提交（临界区只做短操作）。
    inner: RwLock<Inner>,
    /// 片级读写锁：扫第 i 片时只挡第 i 片的写，compact/convert 才拿全部。
    slabs: Box<[RwLock<()>]>,
    slab_rows: u32,
}

/// 唯一的派生表示：`[i8; dim][f32 inv_scale]` 每行。
struct Quant {
    stride: usize,
    file: File,
    map: Arc<Mmap>,
}

/// 写入一律走 pwrite，映射只读：mmap 写会让 FlushFileBuffers 重刷整个视图，
/// 代价随文件增长（800 MB 文件上单次 append 从 16 ms 涨到 34 ms）。
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
    alive: AtomicU32,
}

impl Inner {
    fn dim(&self) -> usize {
        self.meta.dim as usize
    }

    /// 下一个可追加槽位。
    fn n(&self) -> u32 {
        self.meta.watermark()
    }

    fn capacity(&self) -> u32 {
        self.meta.capacity()
    }

    fn slab_count(&self) -> u32 {
        self.meta.slabs.len() as u32
    }

    /// 真正被分配的行数（各片行数之和）。片级压实会在片尾留洞，
    /// 所以它 ≤ 水位 `n()`；`n()` 只表示"下一个可追加槽位"。
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

    /// 整份文件的行，按 f32 视图（含片尾垃圾区）。
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

    /// 数据落盘。映射是只读的，所以只有 pwrite 产生脏页，刷的是增量。
    fn commit(&self) -> Result<()> {
        self.file.sync_all()?;
        if let Some(q) = &self.quant {
            q.file.sync_all()?;
        }
        Ok(())
    }

    /// 首次调用扫一遍首位（@1M 约 10~30 ms），之后靠 append/delete/compact 增量维护。
    fn alive_now(&self) -> u32 {
        let a = self.alive.load(Ordering::Relaxed);
        if a != ALIVE_UNKNOWN {
            return a;
        }
        let dim = self.dim();
        let mut dead = 0u32;
        for slab in 0..self.slab_count() {
            let (lo, hi) = self.slab_range(slab);
            dead += self.rows_all()[lo as usize * dim..hi as usize * dim]
                .chunks_exact(dim)
                .filter(|r| r[0] == 0.0)
                .count() as u32;
        }
        let alive = self.rows_allocated() - dead;
        self.alive.store(alive, Ordering::Relaxed);
        alive
    }

    /// §5.1 AUTO：SQ8 已物化且数据够多才走 SQ8，否则 f32。
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

/// 按整片扩容：加到能容纳 `need` 行为止。
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

/// 校验 + L2 归一化。全部校验通过才返回，调用方不会写出半个批次。
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
        // 墓碑判定是 v[0] == 0，活跃行必须首位非零；扰动在 f32 精度内不可见。
        if dst[0] == 0.0 {
            dst[0] = 1e-30;
        }
    }
    Ok(out)
}

/// 片级压实的回滚：`compacting = Some(slab)` 且备份在 → 把原片写回并清标记；
/// 否则残留的备份都是"已提交但没删掉"，直接删。
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
        // 派生文件可能写到一半，无法逐片修复 → 整体作废，退回纯 f32
        if meta.quant_version.is_some() {
            meta.quant_version = None;
        }
        meta.compacting = None;
        meta.write(dir)?;
    }
    // 清掉所有残留备份（含已提交的）
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

/// 每次编码/写入的行数，既是内存上界也是进度与取消的响应粒度。
const JOB_CHUNK: u32 = 4096;

/// 从 f32 权威数据整份重建派生文件，`meta.quant_version` 是提交点。
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

    // Windows：目标有活动映射时 rename 覆盖会失败，先丢映射。
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
        // 片大小在 create 时定死并写进 meta（它决定槽位 → 行的映射）。
        let slab_rows = cfg
            .slab_rows
            .unwrap_or_else(|| (cfg.initial_capacity / 16).clamp(4096, 1 << 20));
        // 按宿主的 initial_capacity 预分配整片（不足一片也算一片）
        let n_slabs = cfg.initial_capacity.max(1).div_ceil(slab_rows).max(1);
        let meta = Meta {
            format: FORMAT.to_string(),
            dim: cfg.dim,
            slab_rows,
            slabs: vec![0; n_slabs as usize],
            model_hash: cfg.model_hash,
            quant_version: None,
            compacting: None,
        };
        let capacity = meta.capacity();
        // 数据文件先就位，meta 最后写（提交点）：中途崩溃只会留下"没创建"的目录。
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
                alive: AtomicU32::new(0),
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
            // 文件比 meta 记录的长：多出来的是上次扩容没提交的尾巴，截掉
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
            alive: AtomicU32::new(ALIVE_UNKNOWN),
        };

        // 派生表示的提交点是 meta.quant_version：版本不符或没有标记都清掉。
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
        }
        inner.journal.truncate()?;
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
        let normalized = normalize(vecs, dim)?; // 编码在锁外
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
                continue; // 期间被别人追加了，重来
            }
            let need = start.checked_add(m).ok_or(Error::InvalidConfig)?;
            if need > inner.capacity() {
                grow(&mut inner, need)?;
            }
            inner.write_f32_rows(start, &normalized)?;
            inner.commit()?; // 数据先落盘
            // 提交点：把新行记进所属片（一批可能跨多片，前面几片补满）
            let sr = inner.meta.slab_rows;
            let last = (need - 1) / sr;
            for i in 0..=last {
                let lo = i * sr;
                inner.meta.slabs[i as usize] = ((i + 1) * sr).min(need) - lo;
            }
            inner.meta.write(&self.dir)?;

            let a = inner.alive.load(Ordering::Relaxed);
            if a != ALIVE_UNKNOWN {
                inner.alive.store(a + m, Ordering::Relaxed);
            }
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
        inner.journal.commit()?; // ← 提交点
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
            return Ok(()); // 幂等
        }
        let frames: Vec<Frame> = dead.iter().map(|&slot| Frame::Delete { slot }).collect();
        inner.journal.write_batch(&frames)?;
        inner.journal.commit()?; // ← 提交点
        for f in &frames {
            apply_frame(&inner, f)?;
        }
        inner.commit()?;
        inner.journal.truncate()?;

        let a = inner.alive.load(Ordering::Relaxed);
        if a != ALIVE_UNKNOWN {
            inner.alive.store(a - dead.len() as u32, Ordering::Relaxed);
        }
        Ok(())
    }

    /// 压实一片：只重写这一片的区间，其它片一个字节都不动。
    /// 提交点是 meta.slabs[i]；中途崩溃由 `meta.compacting` + `vectors.f32.sN.old` 回滚。
    /// 返回 (old_slots, new_slots) 平行数组（只覆盖本片）；无事可做时返回两个空数组。
    pub fn compact_slab(&self, slab: u32) -> Result<(Vec<u32>, Vec<u32>)> {
        self.compact_slab_with(slab, &JobOptions::default())
    }

    /// 带进度与取消的片级压实。取消（或中途失败）会用备份把这一片写回，
    /// 并作废派生表示——库退化成纯 f32，但不会留下半压实的数据。
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

        // 1) 备份原片（回滚点）
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
        // 2) 标记"正在压实这一片"
        inner.meta.compacting = Some(slab);
        inner.meta.write(&self.dir)?;
        // 3) 分块就地写回。每块都从"原数据"重新读（边写边读会读到刚被清零的片尾），
        //    块之间检查取消。
        let mut wrote = 0u32;
        opts.report(0, alive.len() as u32);
        while wrote < alive.len() as u32 {
            if opts.cancelled() {
                abort_slab(&mut inner, &self.dir, slab, &bak)?;
                return Err(Error::Cancelled);
            }
            let n = (alive.len() as u32 - wrote).min(JOB_CHUNK);
            // 两份缓冲都要在写之前从"原数据"读出来：写下去之后同一段映射看到的就是新数据了。
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
        // 片尾清零（f32 与派生）
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
        // 4) 提交：改片行数并清标记
        inner.meta.slabs[slab as usize] = alive.len() as u32;
        inner.meta.compacting = None;
        inner.meta.write(&self.dir)?;
        // 5) 删除备份
        let _ = fs::remove_file(&bak);

        let a = inner.alive.load(Ordering::Relaxed);
        if a != ALIVE_UNKNOWN {
            inner.alive.store(a, Ordering::Relaxed);
        }
        Ok((alive, (lo..lo + inner.meta.slabs[slab as usize]).collect()))
    }

    /// 逐片压实；每片独立提交（大库建议宿主自己按片调用，好在每片之间重写主键）。
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

    /// 片数与每片行数（宿主按片维护主键时要读）。
    pub fn slab_info(&self) -> (u32, u32) {
        let g = self.inner.read().unwrap();
        (g.slab_count(), g.meta.slab_rows)
    }

    /// 槽位 → 片号；超出 MAX_SLABS 片的部分共用最后一把锁（只是锁变粗，不影响正确性）。
    fn slab_of(&self, slot: u32) -> u32 {
        (slot / self.slab_rows).min(MAX_SLABS as u32 - 1)
    }

    /// 片级读锁：扫第 i 片时只挡第 i 片的写。逐片加锁 → 扫完即放。
    fn read_slab(&self, slab: u32) -> RwLockReadGuard<'_, ()> {
        self.slabs[slab as usize].read().unwrap()
    }

    /// 片级写锁：按片号升序取，和搜索的加锁顺序一致（扫描不取 inner，故无环）。
    fn write_slabs(&self, first: u32, last: u32) -> Vec<RwLockWriteGuard<'_, ()>> {
        (first..=last)
            .map(|i| self.slabs[i as usize].write().unwrap())
            .collect()
    }

    /// 一批槽位覆盖的片区间（用于取片级写锁）。
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
        // 短临界区：只取快照（Arc 克隆），之后全程不持 inner 锁。
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
        let alive = self.inner.read().unwrap().alive_now();

        let actual = if quant.is_some() { Repr::Sq8 } else { Repr::F32 };
        let repr = match opts.repr {
            Some(Repr::F32) => Repr::F32,
            Some(r) if r != actual => actual, // 未物化 → 退回实际可用的
            Some(r) => r,
            None => self.inner.read().unwrap().suggest(alive),
        };
        // 默认重排深度取 2×k：1M 实测 depth=k 只有 96.5~99.2%，depth=2k 是 100%。
        // 显式给出的 rerank 一律照办，不偷偷放大。
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

        // 查询向量量化（每片复用同一份）
        let (qi8, inv_q) = match repr {
            Repr::Sq8 => {
                let (a, b) = quant::encode_query(&qn)?;
                (Some(a), b)
            }
            Repr::F32 => (None, 0.0),
        };

        // 逐片扫描：每片独立加读锁，扫完立刻释放 → 别的片可以同时写。
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
        // 注意：不在这里回写 alive。搜索不持 inner 锁，用快照算出来的 alive 可能已被
        // 并发 delete 改过；alive 只由 append/delete/compact 在 inner 写锁下维护。
        let _ = dead;

        hits.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.slot.cmp(&b.slot))
        });
        hits.truncate(kk);

        // 重排：slots 升序 gather f32，精确点积，再按精确分数排序。
        // 只要走了低精度扫描且请求了重排（depth > 0）就重排，哪怕候选正好只有 k 条——
        // 否则分数会退化成量化值，与"重排后返回精确余弦"的契约不符。
        let do_rerank = repr != Repr::F32 && depth.map_or(false, |d| d > 0);
        let mut ranked: Vec<(f32, u32)> = if do_rerank && !hits.is_empty() {
            // gather 阶段要读 f32，短暂拿全部片读锁（毫秒级）。
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

    /// 最高精度可用表示：f32 恒在。
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

    pub fn suggest_mode(&self) -> Repr {
        let inner = self.inner.read().unwrap();
        inner.suggest(inner.alive_now())
    }

    pub fn convert(&self, target: Repr) -> Result<()> {
        self.convert_with(target, &JobOptions::default())
    }

    /// 带进度与取消的 `convert`。取消时派生文件不落地，库保持原样。
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
            alive: inner.alive_now(),
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

/// 取消/失败时把某一片恢复原状：写回备份、清 `compacting`、作废派生表示。
fn abort_slab(inner: &mut Inner, dir: &Path, slab: u32, bak: &Path) -> Result<()> {
    let stride = inner.stride;
    let (lo, _) = inner.slab_range(slab);
    if bak.exists() {
        let bytes = fs::read(bak)?;
        write_at(&inner.file, &bytes, lo as u64 * stride as u64)?;
        inner.file.sync_all()?;
        let _ = fs::remove_file(bak);
    }
    // 派生文件可能只写了一半，无法逐片修复 → 整体作废
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
