use std::ops::Mul;

use ndarray::parallel::par_azip;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Axis, Zip};
use numpy::PyArray2;

use numpy::ToPyArray;
use numpy::PyUntypedArrayMethods;

use crate::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cpa_update_r(
    _py: Python<'_>,
    p: &ThreadPool,                  // thread pool to run in parallel
    mut ex: PyReadwriteArray1<f64>,  // samples accumulator with shape (nb_samples,)
    mut ex2: PyReadwriteArray1<f64>, // squared samples accumulator with shape (nb_samples,)
    mut ey: PyReadwriteArray1<f64>,  // intermediate data accumulator with shape (nb_data,)
    mut ey2: PyReadwriteArray1<f64>, // squared intermediate data accumulator with shape (nb_data,)
    mut exy: PyReadwriteArray2<f64>, // dot product (intermediate data, samples) accumulator with shape (nb_data, nb_samples)
    x: PyReadonlyArray2<f64>,        // samples to update with shape (batch_size, nb_samples)
    y: PyReadonlyArray2<f64>,        // intermediate data to update with shape (batch_size, nb_data)
) {
    assert_eq!(ex.shape()[0], x.shape()[1]);
    assert_eq!(ey.shape()[0], y.shape()[1]);
    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut ey_r = ey.as_array_mut();
    let mut ey2_r = ey2.as_array_mut();
    let mut exy_r = exy.as_array_mut();
    let x_r = &x.as_array();
    let y_r = &y.as_array();
    // compute sum xy
    let mut _dot = y_r.t().dot(x_r);
    p.on_worker(_py, || {
        // compute sum x and sum x^2
        (
            x_r.axis_iter(Axis(1)),
            ex_r.axis_iter_mut(Axis(0)),
            ex2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(_x, mut _ex, mut _ex2)| {
                let sx = _x.sum();
                let sx2 = _x.mapv(|a: f64| a * a).sum();
                _ex.map_inplace(|a: &mut f64| *a += sx);
                _ex2.map_inplace(|a| *a += sx2);
            });
        // compute sum y and sum y^2
        (
            y_r.axis_iter(Axis(1)),
            ey_r.axis_iter_mut(Axis(0)),
            ey2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(_y, mut _ey, mut _ey2)| {
                let sy = _y.sum();
                let sy2 = _y.mapv(|a| a.powi(2)).sum();
                _ey.map_inplace(|a: &mut f64| *a += sy);
                _ey2.map_inplace(|a| *a += sy2);
            });

        // compute sum xy
        Zip::from(&mut exy_r).and(&_dot).par_for_each(|a, b| {
            *a += b;
        })
    })
}

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cpa_final_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,             // thread pool to run in parallel
    ex: PyReadonlyArray1<f64>,  // samples accumulator with shape (nb_samples,)
    ex2: PyReadonlyArray1<f64>, // squared samples accumulator with shape (nb_samples,)
    ey: PyReadonlyArray1<f64>,  // intermediate data accumulator with shape (nb_data,)
    ey2: PyReadonlyArray1<f64>, // squared intermediate data accumulator with shape (nb_data,)
    exy: PyReadonlyArray2<f64>, // dot product (intermediate data, samples) accumulator with shape (nb_data, nb_samples)
    processed_traces: usize,
) -> Bound<'py, PyArray2<f64>> {
    let ex_r = ex.as_array();
    let ex2_r = ex2.as_array();
    let ey_r = ey.as_array();
    let ey2_r = ey2.as_array();
    let exy_r = exy.as_array();
    let mut sigma_x = Array1::<f64>::zeros(ex_r.dim());
    let mut sigma_y = Array1::<f64>::zeros(ey_r.dim());
    let num: f64 = processed_traces as f64;
    let mut e = exy_r.mul(num) - ey_r.insert_axis(Axis(1)).dot(&ex_r.insert_axis(Axis(0)));

    p.on_worker(_py, || {
        par_azip!((s in &mut sigma_x, x in &ex_r, x2 in &ex2_r) {
            *s = (num * (*x2) - ((*x)*(*x))).sqrt();
        });
        par_azip!((s in &mut sigma_y, y in &ey_r, y2 in &ey2_r) {
            *s = (num * (*y2) - ((*y)*(*y))).sqrt();
        });
        let d = sigma_y
            .insert_axis(Axis(1))
            .dot(&sigma_x.insert_axis(Axis(0)));
        Zip::from(&mut e).and(&d).par_for_each(|a, &b| {
            *a /= b;
            if (*a).is_nan() {
                *a = 0.0;
            }
        });
    });

    return e.to_pyarray_bound(_py);
}


#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cpa_update_r32(
    _py: Python<'_>,
    p: &ThreadPool,                  // thread pool to run in parallel
    mut ex: PyReadwriteArray1<f32>,  // samples accumulator with shape (nb_samples,)
    mut ex2: PyReadwriteArray1<f32>, // squared samples accumulator with shape (nb_samples,)
    mut ey: PyReadwriteArray1<f32>,  // intermediate data accumulator with shape (nb_data,)
    mut ey2: PyReadwriteArray1<f32>, // squared intermediate data accumulator with shape (nb_data,)
    mut exy: PyReadwriteArray2<f32>, // dot product (intermediate data, samples) accumulator with shape (nb_data, nb_samples)
    x: PyReadonlyArray2<f32>,        // samples to update with shape (batch_size, nb_samples)
    y: PyReadonlyArray2<f32>,        // intermediate data to update with shape (batch_size, nb_data)
) {
    assert_eq!(ex.shape()[0], x.shape()[1]);
    assert_eq!(ey.shape()[0], y.shape()[1]);
    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut ey_r = ey.as_array_mut();
    let mut ey2_r = ey2.as_array_mut();
    let mut exy_r = exy.as_array_mut();
    let x_r = &x.as_array();
    let y_r = &y.as_array();
    // compute sum xy
    let mut _dot = y_r.t().dot(x_r);
    p.on_worker(_py, || {
        // compute sum x and sum x^2
        (
            x_r.axis_iter(Axis(1)),
            ex_r.axis_iter_mut(Axis(0)),
            ex2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(_x, mut _ex, mut _ex2)| {
                let sx = _x.sum();
                let sx2 = _x.mapv(|a: f32| a * a).sum();
                _ex.map_inplace(|a: &mut f32| *a += sx);
                _ex2.map_inplace(|a| *a += sx2);
            });
        // compute sum y and sum y^2
        (
            y_r.axis_iter(Axis(1)),
            ey_r.axis_iter_mut(Axis(0)),
            ey2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(_y, mut _ey, mut _ey2)| {
                let sy = _y.sum();
                let sy2 = _y.mapv(|a| a.powi(2)).sum();
                _ey.map_inplace(|a: &mut f32| *a += sy);
                _ey2.map_inplace(|a| *a += sy2);
            });

        // compute sum xy
        Zip::from(&mut exy_r).and(&_dot).par_for_each(|a, b| {
            *a += b;
        })
    })
}

// CPA Distinguisher update function
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cpa_final_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool,             // thread pool to run in parallel
    ex: PyReadonlyArray1<f32>,  // samples accumulator with shape (nb_samples,)
    ex2: PyReadonlyArray1<f32>, // squared samples accumulator with shape (nb_samples,)
    ey: PyReadonlyArray1<f32>,  // intermediate data accumulator with shape (nb_data,)
    ey2: PyReadonlyArray1<f32>, // squared intermediate data accumulator with shape (nb_data,)
    exy: PyReadonlyArray2<f32>, // dot product (intermediate data, samples) accumulator with shape (nb_data, nb_samples)
    processed_traces: usize,
) -> Bound<'py, PyArray2<f32>> {
    let ex_r = ex.as_array();
    let ex2_r = ex2.as_array();
    let ey_r = ey.as_array();
    let ey2_r = ey2.as_array();
    let exy_r = exy.as_array();
    let mut sigma_x = Array1::<f32>::zeros(ex_r.dim());
    let mut sigma_y = Array1::<f32>::zeros(ey_r.dim());
    let num: f32 = processed_traces as f32;
    let mut e = exy_r.mul(num) - ey_r.insert_axis(Axis(1)).dot(&ex_r.insert_axis(Axis(0)));

    p.on_worker(_py, || {
        par_azip!((s in &mut sigma_x, x in &ex_r, x2 in &ex2_r) {
            *s = (num * (*x2) - ((*x)*(*x))).sqrt();
        });
        par_azip!((s in &mut sigma_y, y in &ey_r, y2 in &ey2_r) {
            *s = (num * (*y2) - ((*y)*(*y))).sqrt();
        });
        let d = sigma_y
            .insert_axis(Axis(1))
            .dot(&sigma_x.insert_axis(Axis(0)));
        Zip::from(&mut e).and(&d).par_for_each(|a, &b| {
            *a /= b;
            if (*a).is_nan() {
                *a = 0.0;
            }
        });
    });

    return e.to_pyarray_bound(_py);
}
