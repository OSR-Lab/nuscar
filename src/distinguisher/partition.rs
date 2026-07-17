use ndarray::parallel::prelude::*;
use ndarray::{Axis, Zip};
use numpy::PyReadwriteArray3;
use numpy::PyUntypedArrayMethods;

use crate::*;
use numpy::{PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

// template update function for train
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn partition_update_r(
    _py: Python<'_>,
    p: &ThreadPool,                      // thread pool to run in parallel
    mut ex: PyReadwriteArray3<f64>, // samples accumulator with shape (nb_data, nb_samples, n_partitions)
    mut ex2: PyReadwriteArray3<f64>, // samples squre accumulator with shape (nb_data, nb_samples, n_partitions)
    mut counter: PyReadwriteArray2<u32>, // counter for every partition.(nb_data, n_partitions)
    x: PyReadonlyArray2<f64>,        // samples to update with shape (batch_size, nb_samples)
    d: PyReadonlyArray2<u8>,         // intermediate data to update with shape (batch_size, nb_data)
) {
    let nb_data = ex.shape()[0];
    assert_eq!(nb_data, ex2.shape()[0]);
    assert_eq!(ex.shape()[1], ex2.shape()[1]);
    assert_eq!(nb_data, counter.shape()[0]);
    assert_eq!(ex.shape()[1], x.shape()[1]);
    assert_eq!(nb_data, d.shape()[1]);
    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut counter_r = counter.as_array_mut();

    let x_r = &x.as_array();
    let d_r = &d.as_array();

    p.on_worker(_py, || {
        (
            counter_r.axis_iter_mut(Axis(0)),
            d_r.axis_iter(Axis(1)),
            ex_r.axis_iter_mut(Axis(0)),
            ex2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(mut counter_d, di, mut exi, mut ex2i)| {
                Zip::from(x_r.axis_iter(Axis(0)))
                    .and(di)
                    .for_each(|x_i, dij| {
                        let partition = *dij as usize;
                        counter_d[partition] += 1;
                        Zip::from(exi.index_axis_mut(Axis(1), partition))
                            .and(ex2i.index_axis_mut(Axis(1), partition))
                            .and(x_i)
                            .into_par_iter()
                            .for_each(|(exij, ex2ij, s)| {
                                *exij += *s;
                                *ex2ij += (*s) * (*s);
                            })
                    })
            })
    })
}

// template update function for train
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn partition_update_r32(
    _py: Python<'_>,
    p: &ThreadPool,                      // thread pool to run in parallel
    mut ex: PyReadwriteArray3<f32>, // samples accumulator with shape (nb_data, nb_samples, n_partitions)
    mut ex2: PyReadwriteArray3<f32>, // samples squre accumulator with shape (nb_data, nb_samples, n_partitions)
    mut counter: PyReadwriteArray2<u32>, // counter for every partition.(nb_data, n_partitions)
    x: PyReadonlyArray2<f32>,        // samples to update with shape (batch_size, nb_samples)
    d: PyReadonlyArray2<u8>,         // intermediate data to update with shape (batch_size, nb_data)
) {
    let nb_data = ex.shape()[0];
    assert_eq!(nb_data, ex2.shape()[0]);
    assert_eq!(ex.shape()[1], ex2.shape()[1]);
    assert_eq!(nb_data, counter.shape()[0]);
    assert_eq!(ex.shape()[1], x.shape()[1]);
    assert_eq!(nb_data, d.shape()[1]);
    let mut ex_r = ex.as_array_mut();
    let mut ex2_r = ex2.as_array_mut();
    let mut counter_r = counter.as_array_mut();

    let x_r = &x.as_array();
    let d_r = &d.as_array();

    p.on_worker(_py, || {
        (
            counter_r.axis_iter_mut(Axis(0)),
            d_r.axis_iter(Axis(1)),
            ex_r.axis_iter_mut(Axis(0)),
            ex2_r.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(mut counter_d, di, mut exi, mut ex2i)| {
                Zip::from(x_r.axis_iter(Axis(0)))
                    .and(di)
                    .for_each(|x_i, dij| {
                        let partition = *dij as usize;
                        counter_d[partition] += 1;
                        Zip::from(exi.index_axis_mut(Axis(1), partition))
                            .and(ex2i.index_axis_mut(Axis(1), partition))
                            .and(x_i)
                            .into_par_iter()
                            .for_each(|(exij, ex2ij, s)| {
                                *exij += *s;
                                *ex2ij += (*s) * (*s);
                            })
                    })
            })
    })
}
