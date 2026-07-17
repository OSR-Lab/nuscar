use ndarray::parallel::prelude::*;
use ndarray::s;

use ndarray::{Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use ndarray_conv::*;
use numpy::ToPyArray;
use numpy::PyUntypedArrayMethods;

use crate::*;

#[pyfunction]
pub fn pattern_corr_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f64>,
    pattern: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let pattern_r = pattern.as_array();
    let (out_row, mut out_col) = sample_r.dim();
    let window_size = pattern_r.len();
    out_col = out_col - window_size + 1;
    // let pattern_reverse_r = pattern_r.slice(s![..;-1]).insert_axis(Axis(0));
    let mut output = Array2::<f64>::zeros((out_row, out_col));
    let sx = pattern_r.sum();
    let sx2 = pattern_r.mapv(|a: f64| a * a).sum();
    let num = pattern_r.len() as f64;

    let sigma_x = (num * sx2 - sx * sx).sqrt();
    let pattern_r_num = (pattern_r.mapv(|a| a * num)).insert_axis(Axis(0));
    let sxy = sample_r
        .conv_2d_fft(&pattern_r_num, PaddingSize::Valid, PaddingMode::Zeros)
        .unwrap();

    p.on_worker(_py, || {
        (
            sample_r.axis_iter(Axis(0)),
            sxy.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(row_sample, xy_row, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sy = slice.sum();
                let mut sy2 = slice.mapv(|a: f64| a * a).sum();

                for (i, v) in row_out.indexed_iter_mut() {
                    sy += row_sample[i + window_size - 1];
                    sy2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    let sigma_y = (num * sy2 - sy * sy).sqrt();
                    *v = (xy_row[i] - sy * sx) / (sigma_x * sigma_y);
                    sy -= row_sample[i];
                    sy2 -= row_sample[i] * row_sample[i];
                }
            });
    });
    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn pattern_dist_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f64>,
    pattern: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let pattern_r = pattern.as_array().insert_axis(Axis(0));
    let (out_row, mut out_col) = sample_r.dim();
    let window_size = pattern.len();
    out_col = out_col - window_size + 1;
    // let pattern_reverse_r = pattern_r.slice(s![..;-1]).insert_axis(Axis(0));
    let mut output = Array2::<f64>::zeros((out_row, out_col));
    let sx2 = pattern_r.mapv(|a: f64| a * a).sum();
    let mut sxy = sample_r
        .conv_2d_fft(&pattern_r, PaddingSize::Valid, PaddingMode::Zeros)
        .unwrap();

    p.on_worker(_py, || {
        sxy.par_mapv_inplace(|a| (-2.0 * a)); // -2*xy

        (
            sample_r.axis_iter(Axis(0)),
            sxy.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(row_sample, xy_row, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sy2 = slice.mapv(|a: f64| a * a).sum();

                for (i, v) in row_out.indexed_iter_mut() {
                    sy2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    *v = (sx2 + xy_row[i] + sy2).sqrt();
                    sy2 -= row_sample[i] * row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            if (*a).is_nan() {
                *a = 0.0;
            }
        })
    });

    return output.to_pyarray_bound(_py);
}


#[pyfunction]
pub fn pattern_corr_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f32>,
    pattern: PyReadonlyArray1<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let sample_r = samples.as_array();
    let pattern_r = pattern.as_array();
    let (out_row, mut out_col) = sample_r.dim();
    let window_size = pattern_r.len();
    out_col = out_col - window_size + 1;
    // let pattern_reverse_r = pattern_r.slice(s![..;-1]).insert_axis(Axis(0));
    let mut output = Array2::<f32>::zeros((out_row, out_col));
    let sx = pattern_r.sum();
    let sx2 = pattern_r.mapv(|a: f32| a * a).sum();
    let num = pattern_r.len() as f32;

    let sigma_x = (num * sx2 - sx * sx).sqrt();
    let pattern_r_num = (pattern_r.mapv(|a| a * num)).insert_axis(Axis(0));
    let sxy = sample_r
        .conv_2d_fft(&pattern_r_num, PaddingSize::Valid, PaddingMode::Zeros)
        .unwrap();

    p.on_worker(_py, || {
        (
            sample_r.axis_iter(Axis(0)),
            sxy.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(row_sample, xy_row, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sy = slice.sum();
                let mut sy2 = slice.mapv(|a: f32| a * a).sum();

                for (i, v) in row_out.indexed_iter_mut() {
                    sy += row_sample[i + window_size - 1];
                    sy2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    let sigma_y = (num * sy2 - sy * sy).sqrt();
                    *v = (xy_row[i] - sy * sx) / (sigma_x * sigma_y);
                    sy -= row_sample[i];
                    sy2 -= row_sample[i] * row_sample[i];
                }
            });
    });
    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn pattern_dist_r32<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    samples: PyReadonlyArray2<f32>,
    pattern: PyReadonlyArray1<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let sample_r = samples.as_array();
    let pattern_r = pattern.as_array().insert_axis(Axis(0));
    let (out_row, mut out_col) = sample_r.dim();
    let window_size = pattern.len();
    out_col = out_col - window_size + 1;
    // let pattern_reverse_r = pattern_r.slice(s![..;-1]).insert_axis(Axis(0));
    let mut output = Array2::<f32>::zeros((out_row, out_col));
    let sx2 = pattern_r.mapv(|a: f32| a * a).sum();
    let mut sxy = sample_r
        .conv_2d_fft(&pattern_r, PaddingSize::Valid, PaddingMode::Zeros)
        .unwrap();

    p.on_worker(_py, || {
        sxy.par_mapv_inplace(|a| (-2.0 * a)); // -2*xy

        (
            sample_r.axis_iter(Axis(0)),
            sxy.axis_iter(Axis(0)),
            output.axis_iter_mut(Axis(0)),
        )
            .into_par_iter()
            .for_each(|(row_sample, xy_row, mut row_out)| {
                let slice = row_sample.slice(s![0..window_size - 1]);
                let mut sy2 = slice.mapv(|a: f32| a * a).sum();

                for (i, v) in row_out.indexed_iter_mut() {
                    sy2 += row_sample[i + window_size - 1] * row_sample[i + window_size - 1];
                    *v = (sx2 + xy_row[i] + sy2).sqrt();
                    sy2 -= row_sample[i] * row_sample[i];
                }
            });
        output.par_map_inplace(|a| {
            if (*a).is_nan() {
                *a = 0.0;
            }
        })
    });

    return output.to_pyarray_bound(_py);
}
