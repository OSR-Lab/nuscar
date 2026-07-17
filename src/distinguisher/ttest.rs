use ndarray::parallel::par_azip;
use ndarray::{Array, Zip};
use numpy::PyArray1;

use numpy::ToPyArray;
use numpy::PyUntypedArrayMethods;

use crate::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ttest_update_r(
    _py: Python<'_>,
    p: &ThreadPool,                  // thread pool to run in parallel
    mut ex: PyReadwriteArray1<f64>,  // set 0 samples accumulator with shape (nb_samples,)
    mut ex2: PyReadwriteArray1<f64>, // set 0 squared samples accumulator with shape (nb_samples,)
    mut ey: PyReadwriteArray1<f64>,  // set 1 samples accumulator with shape (nb_samples,)
    mut ey2: PyReadwriteArray1<f64>, // set 1 squared intermediate data accumulator with shape (nb_samples,)
    batch: PyReadonlyArray2<f64>,    // samples to update with shape (batch_size, nb_samples)
    s: PyReadonlyArray2<u32>,        // classification with shape (batch_size)
) -> (usize, usize) {
    assert_eq!(batch.shape()[0], s.shape()[0]);
    assert_eq!(1, s.shape()[1]);

    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut ey_r = ey.as_array_mut();
    let mut ey2_r = ey2.as_array_mut();
    let batch_r = batch.as_array();
    let s_r = s.as_array();
    // compute sum xy
    let mut n0: usize = 0; // number of set0 in the current batch
    let mut n1: usize = 0; // number of set1 in the current batch
    p.on_worker(_py, || {
        Zip::indexed(s_r.column(0)).for_each(|i, t| {
            if *t == 0 {
                n0 += 1;
                Zip::from(&batch_r.row(i))
                    .and(&mut ex_r)
                    .and(&mut ex2_r)
                    .par_for_each(|xi, exi, ex2i| {
                        *exi += xi;
                        *ex2i += xi * xi;
                    })
            }
            if *t == 1 {
                n1 += 1;
                Zip::from(&batch_r.row(i))
                    .and(&mut ey_r)
                    .and(&mut ey2_r)
                    .par_for_each(|yi, eyi, ey2i| {
                        *eyi += yi;
                        *ey2i += yi * yi;
                    })
            }
        })
    });
    (n0, n1)
}

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ttest_final_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,             // thread pool to run in parallel
    ex: PyReadonlyArray1<f64>,  // samples accumulator with shape (nb_samples,)
    ex2: PyReadonlyArray1<f64>, // squared samples accumulator with shape (nb_samples,)
    ey: PyReadonlyArray1<f64>,  // intermediate data accumulator with shape (nb_data,)
    ey2: PyReadonlyArray1<f64>, // squared intermediate data accumulator with shape (nb_data,)
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f64>> {
    let ex_r = ex.as_array();
    let ex2_r = ex2.as_array();
    let ey_r = ey.as_array();
    let ey2_r = ey2.as_array();
    let nx_r: f64 = nx as f64;
    let ny_r: f64 = ny as f64;
    let mut e = Array::zeros(ex_r.raw_dim());
    p.on_worker(_py, || {
        par_azip!((t in &mut e, exi in &ex_r, ex2i in &ex2_r, eyi in &ey_r, ey2i in &ey2_r) {
            let exi_mean = exi/nx_r;
            let eyi_mean = eyi/ny_r;
            let var_x = ex2i/nx_r - exi_mean*exi_mean;
            let var_y = ey2i/ny_r - eyi_mean*eyi_mean;
            *t = (exi_mean - eyi_mean)/((var_x/nx_r + var_y/ny_r).sqrt());
            if (*t).is_nan() {
                *t = 0.0;
            }
        });
    });

    return e.to_pyarray_bound(_py);
}


// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ttest_update_r32(
    _py: Python<'_>,
    p: &ThreadPool,                  // thread pool to run in parallel
    mut ex: PyReadwriteArray1<f32>,  // set 0 samples accumulator with shape (nb_samples,)
    mut ex2: PyReadwriteArray1<f32>, // set 0 squared samples accumulator with shape (nb_samples,)
    mut ey: PyReadwriteArray1<f32>,  // set 1 samples accumulator with shape (nb_samples,)
    mut ey2: PyReadwriteArray1<f32>, // set 1 squared intermediate data accumulator with shape (nb_samples,)
    batch: PyReadonlyArray2<f32>,    // samples to update with shape (batch_size, nb_samples)
    s: PyReadonlyArray2<u32>,        // classification with shape (batch_size)
) -> (usize, usize) {
    assert_eq!(batch.shape()[0], s.shape()[0]);
    assert_eq!(1, s.shape()[1]);

    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut ey_r = ey.as_array_mut();
    let mut ey2_r = ey2.as_array_mut();
    let batch_r = batch.as_array();
    let s_r = s.as_array();
    // compute sum xy
    let mut n0: usize = 0; // number of set0 in the current batch
    let mut n1: usize = 0; // number of set1 in the current batch
    p.on_worker(_py, || {
        Zip::indexed(s_r.column(0)).for_each(|i, t| {
            if *t == 0 {
                n0 += 1;
                Zip::from(&batch_r.row(i))
                    .and(&mut ex_r)
                    .and(&mut ex2_r)
                    .par_for_each(|xi, exi, ex2i| {
                        *exi += xi;
                        *ex2i += xi * xi;
                    })
            }
            if *t == 1 {
                n1 += 1;
                Zip::from(&batch_r.row(i))
                    .and(&mut ey_r)
                    .and(&mut ey2_r)
                    .par_for_each(|yi, eyi, ey2i| {
                        *eyi += yi;
                        *ey2i += yi * yi;
                    })
            }
        })
    });
    (n0, n1)
}

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ttest_final_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool,             // thread pool to run in parallel
    ex: PyReadonlyArray1<f32>,  // samples accumulator with shape (nb_samples,)
    ex2: PyReadonlyArray1<f32>, // squared samples accumulator with shape (nb_samples,)
    ey: PyReadonlyArray1<f32>,  // intermediate data accumulator with shape (nb_data,)
    ey2: PyReadonlyArray1<f32>, // squared intermediate data accumulator with shape (nb_data,)
    nx: usize,
    ny: usize,
) -> Bound<'py, PyArray1<f32>> {
    let ex_r = ex.as_array();
    let ex2_r = ex2.as_array();
    let ey_r = ey.as_array();
    let ey2_r = ey2.as_array();
    let nx_r: f32 = nx as f32;
    let ny_r: f32 = ny as f32;
    let mut e = Array::zeros(ex_r.raw_dim());
    p.on_worker(_py, || {
        par_azip!((t in &mut e, exi in &ex_r, ex2i in &ex2_r, eyi in &ey_r, ey2i in &ey2_r) {
            let exi_mean = exi/nx_r;
            let eyi_mean = eyi/ny_r;
            let var_x = ex2i/nx_r - exi_mean*exi_mean;
            let var_y = ey2i/ny_r - eyi_mean*eyi_mean;
            *t = (exi_mean - eyi_mean)/((var_x/nx_r + var_y/ny_r).sqrt());
            if (*t).is_nan() {
                *t = 0.0;
            }
        });
    });

    return e.to_pyarray_bound(_py);
}
