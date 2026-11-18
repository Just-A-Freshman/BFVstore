use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    NotFound,
    InvalidConfig,
    DimMismatch,
    ModelMismatch,
    InvalidSlot,
    DegenerateVector,
    NonFinite,
    Locked,
    Empty,
    Unsupported,
    Cancelled,
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => f.write_str("store 不存在"),
            Error::InvalidConfig => f.write_str("配置或 meta 非法"),
            Error::DimMismatch => f.write_str("维度不匹配"),
            Error::ModelMismatch => f.write_str("模型指纹不匹配"),
            Error::InvalidSlot => f.write_str("槽位非法或已删除"),
            Error::DegenerateVector => f.write_str("零向量（范数为 0）"),
            Error::NonFinite => f.write_str("向量含 NaN/Inf"),
            Error::Locked => f.write_str("store 已被其他进程占用"),
            Error::Empty => f.write_str("store 为空"),
            Error::Unsupported => f.write_str("该操作尚未实现"),
            Error::Cancelled => f.write_str("操作已取消"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
