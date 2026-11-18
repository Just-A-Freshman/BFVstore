use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::error::Result;

/// 就地变更 WAL：overwrite / delete。帧格式 `[u32 len][u8 kind][u32 slot][row?]`，
/// `len = 0` 为结束帧 —— 无结束帧即整批未提交，重放时整批丢弃。
pub enum Frame {
    Overwrite { slot: u32, row: Vec<f32> },
    Delete { slot: u32 },
}

const KIND_OVERWRITE: u8 = 1;
const KIND_DELETE: u8 = 2;

pub struct Journal {
    file: File,
}

impl Journal {
    pub fn open(path: PathBuf) -> Result<Journal> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        Ok(Journal { file })
    }

    /// 整批写成一段连续字节，尾部带结束帧。
    pub fn write_batch(&mut self, frames: &[Frame]) -> Result<()> {
        let mut buf = Vec::with_capacity(64);
        for f in frames {
            match f {
                Frame::Overwrite { slot, row } => {
                    let len = 1 + 4 + row.len() * 4;
                    buf.extend_from_slice(&(len as u32).to_le_bytes());
                    buf.push(KIND_OVERWRITE);
                    buf.extend_from_slice(&slot.to_le_bytes());
                    for v in row {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Frame::Delete { slot } => {
                    buf.extend_from_slice(&5u32.to_le_bytes());
                    buf.push(KIND_DELETE);
                    buf.extend_from_slice(&slot.to_le_bytes());
                }
            }
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&buf)?;
        self.file.set_len(buf.len() as u64)?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    pub fn truncate(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// 有结束帧 → `Some(整批)`；被截断或为空 → `None`。
    pub fn read_batch(&mut self) -> Result<Option<Vec<Frame>>> {
        let mut data = Vec::new();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_to_end(&mut data)?;
        let mut frames = Vec::new();
        let mut pos = 0usize;
        loop {
            if pos + 4 > data.len() {
                return Ok(None);
            }
            let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if len == 0 {
                return Ok(Some(frames));
            }
            if len < 5 || pos + len > data.len() {
                return Ok(None);
            }
            let payload = &data[pos..pos + len];
            pos += len;
            let slot = u32::from_le_bytes(payload[1..5].try_into().unwrap());
            match payload[0] {
                KIND_OVERWRITE => {
                    let row = payload[5..]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    frames.push(Frame::Overwrite { slot, row });
                }
                KIND_DELETE => frames.push(Frame::Delete { slot }),
                _ => return Ok(None),
            }
        }
    }
}
