use ndarray::Zip;
use ndarray::{Array2, Axis};
use numpy::PyArray2;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::ToPyArray;
use numpy::PyUntypedArrayMethods;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::*;

#[pyfunction]
pub fn combine_product_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f64>,
    frame2: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = f1_s[idx1] * f2_s[idx2];
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_diff_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f64>,
    frame2: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = f1_s[idx1] - f2_s[idx2];
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_abs_diff_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f64>,
    frame2: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = (f1_s[idx1] - f2_s[idx2]).abs();
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_center_product_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f64>,
    frame2: PyReadonlyArray2<f64>,
    mean1: PyReadonlyArray1<f64>,
    mean2: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    assert_eq!(frame1.shape()[1], mean1.shape()[0]);
    assert_eq!(frame2.shape()[1], mean2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let m1_r = mean1.as_array();
    let m2_r = mean2.as_array();

    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = (f1_s[idx1] - m1_r[idx1]) * (f2_s[idx2] - m2_r[idx2]);
                })
            });
    });

    return output.to_pyarray_bound(_py);
}


#[pyfunction]
pub fn combine_product_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f32>,
    frame2: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = f1_s[idx1] * f2_s[idx2];
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_diff_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f32>,
    frame2: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = f1_s[idx1] - f2_s[idx2];
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_abs_diff_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f32>,
    frame2: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = (f1_s[idx1] - f2_s[idx2]).abs();
                })
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn combine_center_product_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    frame1: PyReadonlyArray2<f32>,
    frame2: PyReadonlyArray2<f32>,
    mean1: PyReadonlyArray1<f32>,
    mean2: PyReadonlyArray1<f32>,
) -> Bound<'py, PyArray2<f32>> {
    assert_eq!(frame1.shape()[0], frame2.shape()[0]);
    assert_eq!(frame1.shape()[1], mean1.shape()[0]);
    assert_eq!(frame2.shape()[1], mean2.shape()[0]);
    let f1_r = frame1.as_array();
    let f2_r = frame2.as_array();
    let m1_r = mean1.as_array();
    let m2_r = mean2.as_array();

    let nrow = f1_r.dim().0;
    let ns1 = f1_r.dim().1;
    let ns2 = f2_r.dim().1;
    let out_ncol = ns1 * ns2;
    let mut output = Array2::zeros((nrow, out_ncol));
    p.on_worker(_py, || {
        (
            f1_r.axis_iter(Axis(0)),
            f2_r.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(f1_s, f2_s, row_out)| {
                Zip::indexed(row_out).par_for_each(|idx, o| {
                    let idx1 = idx / ns2;
                    let idx2 = idx % ns2;
                    *o = (f1_s[idx1] - m1_r[idx1]) * (f2_s[idx2] - m2_r[idx2]);
                })
            });
    });

    return output.to_pyarray_bound(_py);
}
