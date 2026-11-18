use std::fs::{self, File};
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const FORMAT: &str = "vstore-l2-3";
pub const META_FILE: &str = "meta.json";
pub const MAX_DIM: u32 = 4096;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Meta {
    pub format: String,
    pub dim: u32,
    pub slab_rows: u32,
    pub slabs: Vec<u32>,
    pub model_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_version: Option<u32>,
    #[serde(default)]
    pub deleted: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compacting: Option<u32>,
}

impl Meta {
    pub fn read(dir: &Path) -> Result<Meta> {
        let data = match fs::read(dir.join(META_FILE)) {
            Ok(d) => d,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(Error::NotFound),
            Err(e) => return Err(Error::Io(e)),
        };
        let m: Meta = serde_json::from_slice(&data).map_err(|_| Error::InvalidConfig)?;
        if m.format != FORMAT
            || m.dim == 0
            || m.dim > MAX_DIM
            || m.slab_rows == 0
            || m.slabs.is_empty()
            || m.slabs.iter().any(|n| *n > m.slab_rows)
        {
            return Err(Error::InvalidConfig);
        }
        Ok(m)
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        let tmp = dir.join("meta.json.tmp");
        {
            let mut f = File::create(&tmp)?;
            serde_json::to_writer(&mut f, self).map_err(|_| Error::InvalidConfig)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, dir.join(META_FILE))?;
        Ok(())
    }

    pub fn capacity(&self) -> u32 {
        self.slabs.len() as u32 * self.slab_rows
    }

    pub fn watermark(&self) -> u32 {
        for (i, n) in self.slabs.iter().enumerate().rev() {
            if *n > 0 {
                return i as u32 * self.slab_rows + n;
            }
        }
        0
    }

    pub fn slab_range(&self, slab: u32) -> (u32, u32) {
        let lo = slab * self.slab_rows;
        (lo, lo + self.slabs[slab as usize])
    }
}
