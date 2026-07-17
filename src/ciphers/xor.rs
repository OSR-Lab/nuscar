use ndarray::{Array, Zip};

use numpy::ToPyArray;

use crate::*;

use numpy::{PyArray3, PyReadonlyArray2};

// Attack selection function
#[pyfunction]
pub fn xor_attack_hw_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    input: PyReadonlyArray2<u8>, // intermediate data accumulator with shape (batchsize, nb_data)
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let input_r = input.as_array();
    let (rows, cols) = input_r.dim();
    let mut res = Array::zeros((rows, guesses.len(), cols));

    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (input_r[[i, k]] ^ guesses[j]) as usize;
            *z = HW_LUT[x];
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn xor_attack_bit_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    input: PyReadonlyArray2<u8>,
    pos: i8,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let input_r = input.as_array();
    let (rows, cols) = input_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = input_r[[i, k]] ^ guesses[j];
            *z = (x >> pos) & 1;
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn xor_attack_value_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    input: PyReadonlyArray2<u8>, // intermediate data accumulator with shape (batchsize, nb_data)
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let input_r = input.as_array();
    let (rows, cols) = input_r.dim();
    let mut res = Array::zeros((rows, guesses.len(), cols));

    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            *z = input_r[[i, k]] ^ guesses[j];
        });
    });
    res.to_pyarray_bound(_py)
}
