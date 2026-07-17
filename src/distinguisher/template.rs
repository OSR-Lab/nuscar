use ndarray::parallel::par_azip;
use ndarray::parallel::prelude::*;
use ndarray::{Axis, Zip};
use numpy::PyReadwriteArray3;
use numpy::PyUntypedArrayMethods;

use crate::*;
use numpy::{PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;

// template update function for train
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn template_update_train_r(
    _py: Python<'_>,
    p: &ThreadPool,                      // thread pool to run in parallel
    mut ex: PyReadwriteArray2<f64>,      // samples accumulator with shape (labels, nb_samples,)
    mut cov: PyReadwriteArray3<f64>,     // covariance with shape (labels, nb_samples, nb_samples)
    mut counter: PyReadwriteArray1<u32>, // counter for every label.
    x: PyReadonlyArray2<f64>,            // samples to update with shape (batch_size, nb_samples)
    d: PyReadonlyArray2<u8>,             // intermediate data to update with shape (batch_size, 1)
) {
    assert_eq!(ex.shape()[1], x.shape()[1]);
    assert_eq!(ex.shape()[0], counter.shape()[0]);
    assert_eq!(ex.shape()[0], cov.shape()[0]);
    let mut ex_r = ex.as_array_mut();
    let mut cov_r = cov.as_array_mut();
    let mut counter_r: ndarray::ArrayBase<ndarray::ViewRepr<&mut u32>, ndarray::Dim<[usize; 1]>> =
        counter.as_array_mut();

    let x_r = &x.as_array();
    let d_r = &d.as_array();

    let labels = counter_r.len();
    let mut labels_ids: Vec<Vec<usize>> = vec![vec![]; labels];
    Zip::indexed(d_r.column(0)).for_each(|i, t| {
        let l = *t as usize;
        labels_ids[l].push(i);
        counter_r[l] += 1;
    });

    p.on_worker(_py, || {
        (
            0..labels,
            ex_r.axis_iter_mut(Axis(0)),
            cov_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(i, mut r, mut cov)| {
                let xi = x_r.select(Axis(0), &labels_ids[i]);
                let exi = xi.sum_axis(Axis(0));
                par_azip!((r in &mut r, &exi in &exi) {*r += exi});
                let covi = xi.t().dot(&xi);
                par_azip!((cov in &mut cov, &covi in &covi) {*cov += covi});
            });
    })
}

// template update function for train
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn template_update_train_r32(
    _py: Python<'_>,
    p: &ThreadPool,                      // thread pool to run in parallel
    mut ex: PyReadwriteArray2<f32>,      // samples accumulator with shape (labels, nb_samples,)
    mut cov: PyReadwriteArray3<f32>,     // covariance with shape (labels, nb_samples, nb_samples)
    mut counter: PyReadwriteArray1<u32>, // counter for every label.
    x: PyReadonlyArray2<f32>,            // samples to update with shape (batch_size, nb_samples)
    d: PyReadonlyArray2<u8>,             // intermediate data to update with shape (batch_size, 1)
) {
    assert_eq!(ex.shape()[1], x.shape()[1]);
    assert_eq!(ex.shape()[0], counter.shape()[0]);
    assert_eq!(ex.shape()[0], cov.shape()[0]);
    let mut ex_r = ex.as_array_mut();
    let mut cov_r = cov.as_array_mut();
    let mut counter_r: ndarray::ArrayBase<ndarray::ViewRepr<&mut u32>, ndarray::Dim<[usize; 1]>> =
        counter.as_array_mut();

    let x_r = &x.as_array();
    let d_r = &d.as_array();

    let labels = counter_r.len();
    let mut labels_ids: Vec<Vec<usize>> = vec![vec![]; labels];
    Zip::indexed(d_r.column(0)).for_each(|i, t| {
        let l = *t as usize;
        labels_ids[l].push(i);
        counter_r[l] += 1;
    });

    p.on_worker(_py, || {
        (
            0..labels,
            ex_r.axis_iter_mut(Axis(0)),
            cov_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(i, mut r, mut cov)| {
                let xi = x_r.select(Axis(0), &labels_ids[i]);
                let exi = xi.sum_axis(Axis(0));
                par_azip!((r in &mut r, &exi in &exi) {*r += exi});
                let covi = xi.t().dot(&xi);
                par_azip!((cov in &mut cov, &covi in &covi) {*cov += covi});
            });
    })
}
