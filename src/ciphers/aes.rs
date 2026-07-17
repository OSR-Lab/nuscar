// AES module with Python bindings and attack functions
use crate::ciphers::aes_common::*;

use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis, Zip};
use numpy::PyArray2;
use numpy::ToPyArray;
use crate::*;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};

// Python binding functions for AES attacks
#[pyfunction]
pub fn aes_attack_first_sbox_hw_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, cols) = plain_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (plain_r[[i, k]] ^ guesses[j]) as usize;
            *z = HW_LUT[SBOX[x] as usize];
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_first_sbox_bit_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    bitpos: i8,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, cols) = plain_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (plain_r[[i, k]] ^ guesses[j]) as usize;
            *z = (SBOX[x] >> bitpos) & 1;
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_first_sbox_value_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, cols) = plain_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = plain_r[[i, k]] ^ guesses[j];
            *z = SBOX[x as usize];
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_last_sbox_hw_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let cipher_r = ciphertext.as_array();
    let (rows, cols) = cipher_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (cipher_r[[i, k]] ^ guesses[j]) as usize;
            *z = HW_LUT[INV_SBOX[x] as usize];
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_last_sbox_bit_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    bitpos: i8,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let cipher_r = ciphertext.as_array();
    let (rows, cols) = cipher_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), cols));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (cipher_r[[i, k]] ^ guesses[j]) as usize;
            *z = (INV_SBOX[x] >> bitpos) & 1;
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_last_round_xor_hw_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    pos_byte: Vec<usize>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let cipher_r = ciphertext.as_array();
    let (rows, _) = cipher_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), pos_byte.len()));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (cipher_r[[i, pos_byte[k]]] ^ guesses[j]) as usize;
            let c = cipher_r[[i, SHIFT_ROWS[pos_byte[k]]]];
            *z = HW_LUT[(INV_SBOX[x] ^ c) as usize];
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_attack_last_round_xor_bit_with_guess_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    bitpos: i8,
    pos_byte: Vec<usize>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let cipher_r = ciphertext.as_array();
    let (rows, _) = cipher_r.dim();

    let mut res: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 3]>> =
        Array::zeros((rows, guesses.len(), pos_byte.len()));
    p.on_worker(_py, || {
        Zip::indexed(&mut res).par_for_each(|(i, j, k), z| {
            let x = (cipher_r[[i, pos_byte[k]]] ^ guesses[j]) as usize;
            let c = cipher_r[[i, SHIFT_ROWS[pos_byte[k]]]];
            *z = ((INV_SBOX[x] ^ c) >> bitpos) & 1;
        });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_encrypt_step_fix_key_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    key: PyReadonlyArray1<u8>,
    at_round: usize,
    at_step: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    match key.len()? {
        16 => {
            let aes = Aes128::new(key.as_slice().unwrap());
            let plaintext_r = plaintext.as_array();
            let mut output = plaintext_r.to_owned();
            let step: Steps = Steps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.encrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        24 => {
            let aes = Aes192::new(key.as_slice().unwrap());
            let plaintext_r = plaintext.as_array();
            let mut output = plaintext_r.to_owned();
            let step: Steps = Steps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.encrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        32 => {
            let aes = Aes256::new(key.as_slice().unwrap());
            let plaintext_r = plaintext.as_array();
            let mut output = plaintext_r.to_owned();
            let step: Steps = Steps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.encrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        _ => {
            panic!("key length error.");
        }
    }
}

#[pyfunction]
pub fn aes_encrypt_step_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    key: PyReadonlyArray2<u8>,
    at_round: usize,
    at_step: u8,
) -> Bound<'py, PyArray2<u8>> {
    let plaintext_r = plaintext.as_array();
    let key_r = key.as_array();
    let mut output = plaintext_r.to_owned();
    let step: Steps = Steps::from(at_step);
    p.on_worker(_py, || {
        (output.axis_iter_mut(Axis(0)), key_r.axis_iter(Axis(0)))
            .into_par_iter()
            .for_each(|(mut p, k)| {
                let tk: &[u8] = k.as_slice().unwrap();
                let input = p.as_slice_mut().unwrap();
                match k.len() {
                    16 => {
                        let aes = Aes128::new(tk);
                        aes.encrypt(input, at_round, step);
                    }
                    24 => {
                        let aes = Aes192::new(tk);
                        aes.encrypt(input, at_round, step);
                    }
                    32 => {
                        let aes = Aes256::new(tk);
                        aes.encrypt(input, at_round, step);
                    }
                    _ => {
                        panic!("key length error.");
                    }
                };
            });
    });
    output.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_decrypt_step_fix_key_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    key: PyReadonlyArray1<u8>,
    at_round: usize,
    at_step: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    match key.len()? {
        16 => {
            let aes = Aes128::new(key.as_slice().unwrap());
            let ciphertext_r = ciphertext.as_array();
            let mut output = ciphertext_r.to_owned();
            let step: InvSteps = InvSteps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.decrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        24 => {
            let aes = Aes192::new(key.as_slice().unwrap());
            let ciphertext_r = ciphertext.as_array();
            let mut output = ciphertext_r.to_owned();
            let step: InvSteps = InvSteps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.decrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        32 => {
            let aes = Aes256::new(key.as_slice().unwrap());
            let ciphertext_r = ciphertext.as_array();
            let mut output = ciphertext_r.to_owned();
            let step: InvSteps = InvSteps::from(at_step);
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut p| {
                        let input = p.as_slice_mut().unwrap();
                        aes.decrypt(input, at_round, step);
                    });
            });
            Ok(output.to_pyarray_bound(_py))
        }
        _ => {
            panic!("key length error.");
        }
    }
}

#[pyfunction]
pub fn aes_decrypt_step_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    ciphertext: PyReadonlyArray2<u8>,
    key: PyReadonlyArray2<u8>,
    at_round: usize,
    at_step: u8,
) -> Bound<'py, PyArray2<u8>> {
    let ciphertext_r = ciphertext.as_array();
    let key_r = key.as_array();
    let mut output = ciphertext_r.to_owned();
    let step: InvSteps = InvSteps::from(at_step);
    p.on_worker(_py, || {
        (output.axis_iter_mut(Axis(0)), key_r.axis_iter(Axis(0)))
            .into_par_iter()
            .for_each(|(mut p, k)| {
                let tk = k.as_slice().unwrap();
                let input = p.as_slice_mut().unwrap();
                match k.len() {
                    16 => {
                        let aes = Aes128::new(tk);
                        aes.decrypt(input, at_round, step);
                    }
                    24 => {
                        let aes = Aes192::new(tk);
                        aes.decrypt(input, at_round, step);
                    }
                    32 => {
                        let aes = Aes256::new(tk);
                        aes.decrypt(input, at_round, step);
                    }
                    _ => {
                        panic!("key length error.");
                    }
                };
            });
    });
    output.to_pyarray_bound(_py)
}

// Platform-specific helper functions
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_helpers {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2,aes")]
    pub unsafe fn sb_sr(block: &mut [u8]) {
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        let zero_k = _mm_setzero_si128();
        m = _mm_aesenclast_si128(m, zero_k);
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
    }

    #[target_feature(enable = "sse2,aes")]
    pub unsafe fn mc(block: &mut [u8]) {
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        let zero_k = _mm_setzero_si128();
        m = _mm_aesdeclast_si128(m, zero_k);
        m = _mm_aesenc_si128(m, zero_k);
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
    }

    #[target_feature(enable = "sse2,aes")]
    pub unsafe fn inv_mc(block: &mut [u8]) {
        let mut m = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        m = _mm_aesimc_si128(m);
        _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, m);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_helpers {
    use core::arch::aarch64::*;

    pub unsafe fn sb_sr(block: &mut [u8]) {
        let mut state = vld1q_u8(block.as_ptr());
        state = vaeseq_u8(state, vdupq_n_u8(0));
        vst1q_u8(block.as_mut_ptr(), state);
    }

    pub unsafe fn mc(block: &mut [u8]) {
        let mut state = vld1q_u8(block.as_ptr());
        state = vaesdq_u8(state, vdupq_n_u8(0));
        state = vaesmcq_u8(vaeseq_u8(state, vdupq_n_u8(0)));
        vst1q_u8(block.as_mut_ptr(), state);
    }

    pub unsafe fn inv_mc(block: &mut [u8]) {
        let state = vld1q_u8(block.as_ptr());
        let result = vaesimcq_u8(state);
        vst1q_u8(block.as_mut_ptr(), result);
    }
}

#[pyfunction]
pub fn aes_sbox_sr_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    data: PyReadonlyArray2<u8>,
) -> Bound<'py, PyArray2<u8>> {
    let data_r = data.as_array();
    let mut output = data_r.to_owned();
    p.on_worker(_py, || {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut p| {
                let d = p.as_slice_mut().unwrap();
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    x86_helpers::sb_sr(d);
                    #[cfg(target_arch = "aarch64")]
                    aarch64_helpers::sb_sr(d);
                }
            });
    });
    output.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_mixcolumns_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    data: PyReadonlyArray2<u8>,
) -> Bound<'py, PyArray2<u8>> {
    let data_r = data.as_array();
    let mut output = data_r.to_owned();
    p.on_worker(_py, || {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut p| {
                let d = p.as_slice_mut().unwrap();
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    x86_helpers::mc(d);
                    #[cfg(target_arch = "aarch64")]
                    aarch64_helpers::mc(d);
                }
            });
    });
    output.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn aes_inv_mixcolumns_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    data: PyReadonlyArray2<u8>,
) -> Bound<'py, PyArray2<u8>> {
    let data_r = data.as_array();
    let mut output = data_r.to_owned();
    p.on_worker(_py, || {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut p| {
                let d = p.as_slice_mut().unwrap();
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    x86_helpers::inv_mc(d);
                    #[cfg(target_arch = "aarch64")]
                    aarch64_helpers::inv_mc(d);
                }
            });
    });
    output.to_pyarray_bound(_py)
}