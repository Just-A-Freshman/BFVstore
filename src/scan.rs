use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AOrd};

use simsimd::SpatialSimilarity;

use crate::quant::inv_scale_of;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub score: f32,
    pub slot: u32,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    slot: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.slot == o.slot
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    /// 堆顶 = 最差：分数低者更差；同分时 slot 大者更差（tie-break 取小 slot）。
    fn cmp(&self, o: &Self) -> Ordering {
        o.score
            .total_cmp(&self.score)
            .then_with(|| self.slot.cmp(&o.slot))
    }
}

/// 常见路径只有一次 `score > cutoff` 比较；堆只在真的更优时才动。
#[inline]
fn offer(heap: &mut BinaryHeap<Cand>, cap: usize, cutoff: &mut f32, c: Cand) {
    if heap.len() < cap {
        heap.push(c);
        if heap.len() == cap {
            *cutoff = heap.peek().map(|w| w.score).unwrap_or(f32::NEG_INFINITY);
        }
        return;
    }
    if c.score > *cutoff {
        heap.pop();
        heap.push(c);
        *cutoff = heap.peek().map(|w| w.score).unwrap_or(f32::NEG_INFINITY);
    } else if c.score == *cutoff {
        if let Some(w) = heap.peek() {
            if c.slot < w.slot {
                heap.pop();
                heap.push(c);
            }
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or(0.0) as f32
}

/// 并行扫描骨架：`score(slot) = None` 表示墓碑，`Some(分数)` 进堆。
/// 返回（分数降序 / slot 升序 的前 kk 条, 墓碑数）。
fn parallel<F>(rows: usize, kk: usize, threads: usize, base_slot: u32, score: F) -> (Vec<Hit>, u32)
where
    F: Fn(usize) -> Option<f32> + Sync,
{
    if rows == 0 || kk == 0 {
        return (Vec::new(), 0);
    }
    const BLOCK: usize = 1024;
    let nblocks = rows.div_ceil(BLOCK);
    let threads = threads.clamp(1, nblocks);
    let next = AtomicUsize::new(0);

    let (hits, dead) = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let score = &score;
                let next = &next;
                s.spawn(move || {
                    let mut heap = BinaryHeap::with_capacity(kk.min(8192));
                    let mut cutoff = f32::NEG_INFINITY;
                    let mut dead = 0u32;
                    loop {
                        let b = next.fetch_add(1, AOrd::Relaxed);
                        if b >= nblocks {
                            break;
                        }
                        let lo = b * BLOCK;
                        let hi = (lo + BLOCK).min(rows);
                        for slot in lo..hi {
                            match score(slot) {
                                None => dead += 1,
                                Some(sc) => offer(
                                    &mut heap,
                                    kk,
                                    &mut cutoff,
                                    Cand {
                                        score: sc,
                                        slot: base_slot + slot as u32,
                                    },
                                ),
                            }
                        }
                    }
                    (heap, dead)
                })
            })
            .collect();

        let mut hits = Vec::new();
        let mut dead = 0u32;
        for h in handles {
            let (heap, d) = h.join().unwrap();
            dead += d;
            hits.extend(heap.into_iter().map(|c| Hit {
                score: c.score,
                slot: c.slot,
            }));
        }
        (hits, dead)
    });

    let mut hits = hits;
    hits.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.slot.cmp(&b.slot))
    });
    hits.truncate(kk);
    (hits, dead)
}

/// F32：行首位为 0 即墓碑。
pub fn top_k(
    base: &[f32],
    dim: usize,
    rows: usize,
    q: &[f32],
    kk: usize,
    threads: usize,
    base_slot: u32,
) -> (Vec<Hit>, u32) {
    parallel(rows, kk, threads, base_slot, |slot| {
        let row = &base[slot * dim..slot * dim + dim];
        if row[0] == 0.0 {
            None
        } else {
            Some(dot(row, q))
        }
    })
}

/// SQ8：`inv_scale == 0` 即墓碑；分数 = dot_i8 × inv_scale_i × inv_scale_q（单调，非余弦）。
pub fn top_k_i8(
    base: &[u8],
    stride: usize,
    dim: usize,
    rows: usize,
    q: &[i8],
    inv_scale_q: f32,
    kk: usize,
    threads: usize,
    base_slot: u32,
) -> (Vec<Hit>, u32) {
    parallel(rows, kk, threads, base_slot, |slot| {
        let row = &base[slot * stride..slot * stride + stride];
        let inv = inv_scale_of(row, dim);
        if inv == 0.0 {
            None
        } else {
            let d: f32 = i8::dot(bytemuck::cast_slice(&row[..dim]), q).unwrap_or(0.0) as f32;
            Some(d * inv * inv_scale_q)
        }
    })
}
