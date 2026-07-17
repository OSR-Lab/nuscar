use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis, Zip};
use numpy::PyArray2;
use numpy::PyReadonlyArray2;

use numpy::ToPyArray;

use crate::*;
use iir_filters::filter::DirectForm2Transposed;
use iir_filters::filter::Filter;
use iir_filters::filter_design::butter;
use iir_filters::filter_design::FilterType;
use iir_filters::sos::zpk2sos;

#[pyfunction]
pub fn low_pass_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    order: u32,
    samples: PyReadonlyArray2<f64>,
    sample_rate: f64,
    cutoff: f64,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let mut output = Array2::<f64>::zeros(sample_r.dim());
    let zpk = butter(order, FilterType::LowPass(cutoff), sample_rate).unwrap();
    let sos = zpk2sos(&zpk, None).unwrap();
    p.on_worker(_py, || {
        (sample_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let mut dft2 = DirectForm2Transposed::new(&sos);
                Zip::from(&row_sample)
                    .and(&mut row_out)
                    .for_each(|x, y| *y = dft2.filter(*x))
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn high_pass_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    order: u32,
    samples: PyReadonlyArray2<f64>,
    sample_rate: f64,
    cutoff: f64,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let mut output = Array2::<f64>::zeros(sample_r.dim());
    let zpk = butter(order, FilterType::HighPass(cutoff), sample_rate).unwrap();
    let sos = zpk2sos(&zpk, None).unwrap();
    p.on_worker(_py, || {
        (sample_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let mut dft2 = DirectForm2Transposed::new(&sos);
                Zip::from(&row_sample)
                    .and(&mut row_out)
                    .for_each(|x, y| *y = dft2.filter(*x))
            });
    });

    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn band_pass_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    order: u32,
    samples: PyReadonlyArray2<f64>,
    sample_rate: f64,
    cutoff_l: f64,
    cutoff_h: f64,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let mut output = Array2::<f64>::zeros(sample_r.dim());
    let zpk = butter(order, FilterType::BandPass(cutoff_l, cutoff_h), sample_rate).unwrap();
    let sos = zpk2sos(&zpk, None).unwrap();
    p.on_worker(_py, || {
        (sample_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let mut dft2 = DirectForm2Transposed::new(&sos);
                Zip::from(&row_sample)
                    .and(&mut row_out)
                    .for_each(|x, y| *y = dft2.filter(*x))
            });
    });
    return output.to_pyarray_bound(_py);
}

#[pyfunction]
pub fn band_stop_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool, // thread pool to run in parallel
    order: u32,
    samples: PyReadonlyArray2<f64>,
    sample_rate: f64,
    cutoff_l: f64,
    cutoff_h: f64,
) -> Bound<'py, PyArray2<f64>> {
    let sample_r = samples.as_array();
    let mut output = Array2::<f64>::zeros(sample_r.dim());
    let zpk = butter(order, FilterType::BandStop(cutoff_l, cutoff_h), sample_rate).unwrap();
    let sos = zpk2sos(&zpk, None).unwrap();
    p.on_worker(_py, || {
        (sample_r.axis_iter(Axis(0)), output.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each(|(row_sample, mut row_out)| {
                let mut dft2 = DirectForm2Transposed::new(&sos);
                Zip::from(&row_sample)
                    .and(&mut row_out)
                    .for_each(|x, y| *y = dft2.filter(*x))
            });
    });
    return output.to_pyarray_bound(_py);
}
