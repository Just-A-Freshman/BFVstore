#![cfg(feature = "python")]
//! abi3 wheel 绑定：numpy 只用于 buffer 协议（C 连续），查询逻辑全在 Rust 侧。

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyFileNotFoundError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{Config, Error, JobOptions, JobProgress, OpenOptions, Repr, SearchOptions, Store};

fn err(e: Error) -> PyErr {
    match e {
        Error::NotFound => PyFileNotFoundError::new_err("store 不存在"),
        Error::Unsupported => PyNotImplementedError::new_err("尚未实现"),
        Error::Io(io) => PyRuntimeError::new_err(format!("io: {io}")),
        other => PyValueError::new_err(other.to_string()),
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    F32,
    SQ8,
    AUTO,
}

impl From<Mode> for Repr {
    fn from(m: Mode) -> Repr {
        match m {
            Mode::SQ8 => Repr::Sq8,
            _ => Repr::F32,
        }
    }
}

fn repr_of(r: Repr) -> Mode {
    match r {
        Repr::F32 => Mode::F32,
        Repr::Sq8 => Mode::SQ8,
    }
}

#[pyclass]
struct CancelToken {
    flag: Arc<AtomicBool>,
}

#[pymethods]
impl CancelToken {
    #[new]
    fn new() -> CancelToken {
        CancelToken {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
    #[getter]
    fn cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

#[pyclass(name = "Store")]
struct PyStore {
    inner: Store,
}

#[pymethods]
impl PyStore {
    /// create(path, dim, model_hash, initial_capacity=1024, slab_rows=None)
    /// slab_rows 决定分片大小（槽位 → 行的映射），创建后不可改。
    #[staticmethod]
    #[pyo3(signature = (path, dim, model_hash, initial_capacity = 1024, slab_rows = None))]
    fn create(
        path: &str,
        dim: u32,
        model_hash: &str,
        initial_capacity: u32,
        slab_rows: Option<u32>,
    ) -> PyResult<PyStore> {
        let mut cfg = Config::new(dim, model_hash).with_capacity(initial_capacity);
        if let Some(rows) = slab_rows {
            cfg = cfg.with_slab_rows(rows);
        }
        Store::create(path, cfg)
            .map(|inner| PyStore { inner })
            .map_err(err)
    }

    #[staticmethod]
    fn open(path: &str, expected_model_hash: &str) -> PyResult<PyStore> {
        Store::open(path, OpenOptions::new(expected_model_hash))
            .map(|inner| PyStore { inner })
            .map_err(err)
    }

    fn append<'py>(
        &self,
        py: Python<'py>,
        vecs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let data = vecs.as_slice()?.to_vec();
        let inner = &self.inner;
        let start = py.detach(|| inner.append(&data)).map_err(err)?;
        let m = (data.len() / self.inner.dim() as usize) as u32;
        Ok((start..start + m).collect::<Vec<u32>>().into_pyarray(py))
    }

    fn overwrite(&self, slots: PyReadonlyArray1<'_, u32>, vecs: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let s = slots.as_slice()?.to_vec();
        let data = vecs.as_slice()?.to_vec();
        self.inner.overwrite(&s, &data).map_err(err)
    }

    fn delete(&self, slots: PyReadonlyArray1<'_, u32>) -> PyResult<()> {
        let s = slots.as_slice()?.to_vec();
        self.inner.delete(&s).map_err(err)
    }

    fn compact<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
        let inner = &self.inner;
        let (old, new) = py.detach(|| inner.compact()).map_err(err)?;
        Ok((old.into_pyarray(py), new.into_pyarray(py)))
    }

    #[pyo3(signature = (q, k, mode = None, rerank = None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: usize,
        mode: Option<PyRef<'_, Mode>>,
        rerank: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>, Mode)> {
        let query = q.as_slice()?.to_vec();
        let repr = mode.map(|m| (*m).into());
        let inner = &self.inner;
        let res = py
            .detach(|| {
                inner.search(
                    &query,
                    k,
                    SearchOptions {
                        repr,
                        rerank,
                        threads: None,
                    },
                )
            })
            .map_err(err)?;
        Ok((
            res.slots.into_pyarray(py),
            res.scores.into_pyarray(py),
            repr_of(res.repr),
        ))
    }

    fn vectors<'py>(
        &self,
        py: Python<'py>,
        slots: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let s = slots.as_slice()?.to_vec();
        let dim = self.inner.dim() as usize;
        let inner = &self.inner;
        let data = py.detach(|| inner.vectors(&s)).map_err(err)?;
        let m = data.len() / dim;
        Ok(numpy::ndarray::Array2::from_shape_vec((m, dim), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    #[pyo3(signature = (target, progress = None, cancel = None))]
    fn convert(
        &self,
        py: Python<'_>,
        target: PyRef<'_, Mode>,
        progress: Option<Py<PyAny>>,
        cancel: Option<&CancelToken>,
    ) -> PyResult<()> {
        let target: Repr = (*target).into();
        let cb = progress.map(|f| {
            move |p: JobProgress| {
                Python::attach(|py| {
                    let _ = f.call1(py, (p.done, p.total));
                })
            }
        });
        let flag = cancel.map(|c| c.flag.as_ref());
        let inner = &self.inner;
        py.detach(|| {
            inner.convert_with(
                target,
                &JobOptions {
                    progress: cb.as_ref().map(|f| f as &dyn Fn(JobProgress)),
                    cancel: flag,
                },
            )
        })
        .map_err(err)
    }

    fn model_hash(&self) -> String {
        self.inner.model_hash()
    }

    #[getter]
    fn dim(&self) -> u32 {
        self.inner.dim()
    }

    #[getter]
    fn n(&self) -> u32 {
        self.inner.n()
    }

    fn warm(&self, py: Python<'_>) -> PyResult<u32> {
        let inner = &self.inner;
        py.detach(|| inner.warm()).map_err(err)
    }

    fn suggest_mode(&self) -> Mode {
        repr_of(self.inner.suggest_mode())
    }

    fn slab_info(&self) -> (u32, u32) {
        self.inner.slab_info()
    }

    fn stats(&self) -> (u32, u32, u32, u32, Mode) {
        let s = self.inner.stats();
        (s.dim, s.n, s.alive, s.capacity, repr_of(s.repr))
    }
}

#[pymodule]
fn vstore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    m.add_class::<Mode>()?;
    m.add_class::<CancelToken>()?;
    Ok(())
}
