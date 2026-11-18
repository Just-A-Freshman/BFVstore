use crate::error::{Error, Result};

/// 量化算法版本，变了就 +1：open 会把旧派生文件当垃圾清掉。
pub const QUANT_VERSION: u32 = 1;

/// SQ8 行布局：`[i8; dim][f32 inv_scale]`，stride = dim + 4。
pub fn stride(dim: usize) -> usize {
    dim + 4
}

pub fn inv_scale_of(row: &[u8], dim: usize) -> f32 {
    f32::from_le_bytes(row[dim..dim + 4].try_into().unwrap())
}

/// 单位向量 → 量化行；全零行（墓碑）→ 全零行 + inv_scale = 0。
/// 返回 inv_scale（0 表示墓碑）。
pub fn encode(src: &[f32], dst: &mut [u8]) -> f32 {
    let dim = src.len();
    let mut max = 0f32;
    for &v in src {
        max = max.max(v.abs());
    }
    if max == 0.0 {
        dst.fill(0);
        return 0.0;
    }
    let s = 127.0 / max;
    for (i, &v) in src.iter().enumerate() {
        dst[i] = (v * s).round().clamp(-127.0, 127.0) as i8 as u8;
    }
    let inv = max / 127.0;
    dst[dim..dim + 4].copy_from_slice(&inv.to_le_bytes());
    inv
}

/// 查询向量量化：返回 (q_i8, inv_scale_q = max|q| / 127)。
pub fn encode_query(q: &[f32]) -> Result<(Vec<i8>, f32)> {
    let mut max = 0f32;
    for &v in q {
        max = max.max(v.abs());
    }
    if max == 0.0 {
        return Err(Error::DegenerateVector);
    }
    let s = 127.0 / max;
    let out = q
        .iter()
        .map(|&v| (v * s).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Ok((out, max / 127.0))
}
