use ndarray::s;
use ndarray::{Array2, Axis};
use numpy::PyArray2;
use numpy::PyReadonlyArray2;

use numpy::ToPyArray;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::*;

#[pyfunction]
pub fn moving_mean_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f64>,
    window_size: usize,
) -> Bound<'py, PyArray2<f64>> {
    let samples_r = samples.as_array();
    // let samples_r = samples.extract::<PyReadonlyArray2<f64>>(_py).unwrap();
    let (nrow, ncol) = samples_r.dim();
    let out_ncol = ncol - window_size + 1;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (samples_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sx = slice.sum();
                for (i, v) in row_out.indexed_iter_mut() {
                    sx += row_sample[i + window_size - 1];
                    *v = sx;
                    sx -= row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            *a /= window_size as f64;
        })
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn moving_var_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,                 // thread pool to run in parallel
    samples: PyReadonlyArray2<f64>, // samples to update with shape (batch_size, nb_samples)
    window_size: usize,
) -> Bound<'py, PyArray2<f64>> {
    let samples_r = samples.as_array();
    // let samples_r = samples.extract::<PyReadonlyArray2<f64>>(_py).unwrap();
    let (nrow, ncol) = samples_r.dim();
    let out_ncol = ncol - window_size + 1;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (samples_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sx = slice.sum();
                let mut sx2 = slice.mapv(|a: f64| a * a).sum();
                for (i, v) in row_out.indexed_iter_mut() {
                    sx += row_sample[i + window_size - 1];
                    sx2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    *v = (sx2 * window_size as f64) - (sx * sx);
                    sx -= row_sample[i];
                    sx2 -= row_sample[i] * row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            *a /= (window_size as f64) * (window_size as f64);
        })
    });

    return output.to_pyarray_bound(_py);
}


#[pyfunction]
pub fn moving_mean_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f32>,
    window_size: usize,
) -> Bound<'py, PyArray2<f32>> {
    let samples_r = samples.as_array();
    // let samples_r = samples.extract::<PyReadonlyArray2<f32>>(_py).unwrap();
    let (nrow, ncol) = samples_r.dim();
    let out_ncol = ncol - window_size + 1;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (samples_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sx = slice.sum();
                for (i, v) in row_out.indexed_iter_mut() {
                    sx += row_sample[i + window_size - 1];
                    *v = sx;
                    sx -= row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            *a /= window_size as f32;
        })
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn moving_var_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool,                 // thread pool to run in parallel
    samples: PyReadonlyArray2<f32>, // samples to update with shape (batch_size, nb_samples)
    window_size: usize,
) -> Bound<'py, PyArray2<f32>> {
    let samples_r = samples.as_array();
    // let samples_r = samples.extract::<PyReadonlyArray2<f32>>(_py).unwrap();
    let (nrow, ncol) = samples_r.dim();
    let out_ncol = ncol - window_size + 1;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (samples_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sx = slice.sum();
                let mut sx2 = slice.mapv(|a: f32| a * a).sum();
                for (i, v) in row_out.indexed_iter_mut() {
                    sx += row_sample[i + window_size - 1];
                    sx2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    *v = (sx2 * window_size as f32) - (sx * sx);
                    sx -= row_sample[i];
                    sx2 -= row_sample[i] * row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            *a /= (window_size as f32) * (window_size as f32);
        })
    });

    return output.to_pyarray_bound(_py);
}
