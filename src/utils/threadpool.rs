use pyo3::create_exception;
use pyo3::exceptions;
use pyo3::prelude::*;

create_exception!(nuscar, ThreadPoolError, exceptions::PyOSError);

#[pyclass]
pub struct ThreadPool {
    pub pool: rayon::ThreadPool,
    pub num_threads: usize,
}

#[pymethods]
impl ThreadPool {
    #[new]
    /// Create a new ThreadPool, with a given number of threads
    pub fn new(num_threads: usize) -> PyResult<Self> {
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e: rayon::ThreadPoolBuildError| {
                    ThreadPoolError::new_err(e.to_string())
                })?,
           num_threads,
        })
    }
}

impl ThreadPool {
    pub fn on_worker<OP, R>(&self, py: Python, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        let pool = &self.pool;
        py.allow_threads(|| pool.install(op))
    }
}
