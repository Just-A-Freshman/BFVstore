//! vstore：只认槽位和向量的本地向量库。
//!
//! 用法：
//! ```
//! # use vstore::{Config, OpenOptions, SearchOptions, Store};
//! # let dir = std::env::temp_dir().join("vstore-doc");
//! # let _ = std::fs::remove_dir_all(&dir);
//! let store = Store::create(&dir, Config::new(4, "sha256:demo")).unwrap();
//! let start = store.append(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
//! assert_eq!(start, 0);
//! let r = store.search(&[1.0, 0.0, 0.0, 0.0], 2, SearchOptions::default()).unwrap();
//! assert_eq!(r.slots, vec![0, 1]);
//! drop(store);
//! # let _ = std::fs::remove_dir_all(&dir);
//! ```

mod error;
mod journal;
mod meta;
mod quant;
mod scan;
mod store;

#[cfg(feature = "python")]
mod py;

pub use error::{Error, Result};
pub use store::{
    Config, JobOptions, JobProgress, OpenOptions, Repr, SearchOptions, SearchResult, Stats,
    Store,
};
