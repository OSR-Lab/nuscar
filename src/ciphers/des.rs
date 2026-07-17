// DES module with Python bindings: attack functions and encrypt/decrypt step functions.
// Mirrors the pattern from aes.rs but uses the full DES Feistel pipeline from des_common.
// Function names use camelCase (addRk) to match the Python API naming convention.
#![allow(non_snake_case)]

use crate::ciphers::des_common::*;

use ndarray::parallel::prelude::*;
use ndarray::{Array, Axis};
use numpy::PyArray2;
use numpy::ToPyArray;
use crate::*;
use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};

// ============================================================================
// Internal helper: shared logic for all 16 DES attack functions
// ============================================================================

/// Internal helper that implements the common attack pattern for DES.
///
/// For each trace and each guess value (0..64):
///   1. Create an expanded key where all 16 rounds have all 8 words = guess
///   2. Run DES encrypt or decrypt up to (at_round, after_step)
///   3. Optionally apply Hamming weight
///
/// Output shape: (n_traces, n_guesses, 8)
fn des_attack_helper<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    data: PyReadonlyArray2<u8>,
    guesses: Vec<u8>,
    at_round: usize,
    after_step: u8,
    decrypt: bool,
    apply_hw: bool,
) -> Bound<'py, PyArray3<u8>> {
    let data_r = data.as_array();
    let (rows, _cols) = data_r.dim();
    let n_guesses = guesses.len();
    let out_cols = 8usize;

    let mut res = Array::zeros((rows, n_guesses, out_cols));
    p.on_worker(_py, || {
        res.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row_guesses)| {
                // Extract this trace's data as [u8; 8]
                let mut input = [0u8; 8];
                for b in 0..8 {
                    input[b] = data_r[[i, b]];
                }

                for (j, &guess) in guesses.iter().enumerate() {
                    // Create expanded key: all 16 rounds, all 8 words = guess
                    let expanded_key = [[guess; 8]; 16];

                    let result = if decrypt {
                        des_decrypt_at(&input, &expanded_key, at_round, after_step)
                    } else {
                        des_encrypt_at(&input, &expanded_key, at_round, after_step)
                    };

                    for w in 0..out_cols {
                        if apply_hw {
                            row_guesses[[j, w]] = HW_LUT[result[w] as usize];
                        } else {
                            row_guesses[[j, w]] = result[w];
                        }
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

// ============================================================================
// Macro to generate DES attack #[pyfunction] wrappers
// ============================================================================

macro_rules! des_attack_fn {
    ($name:ident, $step:expr, $decrypt:expr, $apply_hw:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
            guesses: Vec<u8>,
        ) -> Bound<'py, PyArray3<u8>> {
            des_attack_helper(_py, p, data, guesses, 0, $step as u8, $decrypt, $apply_hw)
        }
    };
}

// First-round (encrypt) attack functions
des_attack_fn!(des_attack_first_addRk_hw_with_guess_r,    DesSteps::AddRoundKey,             false, true);
des_attack_fn!(des_attack_first_addRk_value_with_guess_r, DesSteps::AddRoundKey,             false, false);
des_attack_fn!(des_attack_first_sbox_hw_with_guess_r,     DesSteps::Sboxes,                  false, true);
des_attack_fn!(des_attack_first_sbox_value_with_guess_r,  DesSteps::Sboxes,                  false, false);
des_attack_fn!(des_attack_first_round_hw_with_guess_r,    DesSteps::InvPermutationPRight,     false, true);
des_attack_fn!(des_attack_first_round_value_with_guess_r, DesSteps::InvPermutationPRight,     false, false);
des_attack_fn!(des_attack_delta_first_round_hw_with_guess_r,    DesSteps::InvPermutationPDeltaRight, false, true);
des_attack_fn!(des_attack_delta_first_round_value_with_guess_r, DesSteps::InvPermutationPDeltaRight, false, false);

// Last-round (decrypt) attack functions
des_attack_fn!(des_attack_last_addRk_hw_with_guess_r,    DesSteps::AddRoundKey,             true, true);
des_attack_fn!(des_attack_last_addRk_value_with_guess_r, DesSteps::AddRoundKey,             true, false);
des_attack_fn!(des_attack_last_sbox_hw_with_guess_r,     DesSteps::Sboxes,                  true, true);
des_attack_fn!(des_attack_last_sbox_value_with_guess_r,  DesSteps::Sboxes,                  true, false);
des_attack_fn!(des_attack_last_round_hw_with_guess_r,    DesSteps::InvPermutationPRight,     true, true);
des_attack_fn!(des_attack_last_round_value_with_guess_r, DesSteps::InvPermutationPRight,     true, false);
des_attack_fn!(des_attack_delta_last_round_hw_with_guess_r,    DesSteps::InvPermutationPDeltaRight, true, true);
des_attack_fn!(des_attack_delta_last_round_value_with_guess_r, DesSteps::InvPermutationPDeltaRight, true, false);

// ============================================================================
// Macros to generate DES encrypt/decrypt step #[pyfunction] wrappers
// ============================================================================

macro_rules! des_step_fix_key_fn {
    ($name:ident, $cipher_fn:ident) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
            key: PyReadonlyArray1<u8>,
            at_round: usize,
            after_step: u8,
        ) -> Bound<'py, PyArray2<u8>> {
            let key_slice = key.as_slice().unwrap();
            let mut key_arr = [0u8; 8];
            key_arr.copy_from_slice(key_slice);
            let expanded_key = key_schedule(&key_arr);

            let data_r = data.as_array();
            let (rows, _cols) = data_r.dim();
            let mut output = Array::zeros((rows, 8));
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut buf = [0u8; 8];
                        for b in 0..8 {
                            buf[b] = data_r[[i, b]];
                        }
                        let result = $cipher_fn(&buf, &expanded_key, at_round, after_step);
                        for b in 0..8 {
                            row[b] = result[b];
                        }
                    });
            });
            output.to_pyarray_bound(_py)
        }
    };
}

macro_rules! des_step_fn {
    ($name:ident, $cipher_fn:ident) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
            key: PyReadonlyArray2<u8>,
            at_round: usize,
            after_step: u8,
        ) -> Bound<'py, PyArray2<u8>> {
            let data_r = data.as_array();
            let key_r = key.as_array();
            let (rows, _cols) = data_r.dim();
            let mut output = Array::zeros((rows, 8));
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut buf = [0u8; 8];
                        for b in 0..8 {
                            buf[b] = data_r[[i, b]];
                        }
                        let mut key_arr = [0u8; 8];
                        for b in 0..8 {
                            key_arr[b] = key_r[[i, b]];
                        }
                        let expanded_key = key_schedule(&key_arr);
                        let result = $cipher_fn(&buf, &expanded_key, at_round, after_step);
                        for b in 0..8 {
                            row[b] = result[b];
                        }
                    });
            });
            output.to_pyarray_bound(_py)
        }
    };
}

des_step_fix_key_fn!(des_encrypt_step_fix_key_r, des_encrypt_at);
des_step_fn!(des_encrypt_step_r, des_encrypt_at);
des_step_fix_key_fn!(des_decrypt_step_fix_key_r, des_decrypt_at);
des_step_fn!(des_decrypt_step_r, des_decrypt_at);

// ============================================================================
// DES primitive #[pyfunction] wrappers (like AES's aes_mixcolumns_r)
// ============================================================================

/// Macro for primitives with same-size input/output (8→8)
macro_rules! des_prim_8to8_fn {
    ($name:ident, $prim_fn:ident) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
        ) -> Bound<'py, PyArray2<u8>> {
            let data_r = data.as_array();
            let (rows, _cols) = data_r.dim();
            let mut output = Array::zeros((rows, 8));
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut buf = [0u8; 8];
                        for b in 0..8 {
                            buf[b] = data_r[[i, b]];
                        }
                        let result = $prim_fn(&buf);
                        for b in 0..8 {
                            row[b] = result[b];
                        }
                    });
            });
            output.to_pyarray_bound(_py)
        }
    };
}

/// Macro for primitives with 4→8 input/output
macro_rules! des_prim_4to8_fn {
    ($name:ident, $prim_fn:ident) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
        ) -> Bound<'py, PyArray2<u8>> {
            let data_r = data.as_array();
            let (rows, _cols) = data_r.dim();
            let mut output = Array::zeros((rows, 8));
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut buf = [0u8; 4];
                        for b in 0..4 {
                            buf[b] = data_r[[i, b]];
                        }
                        let result = $prim_fn(&buf);
                        for b in 0..8 {
                            row[b] = result[b];
                        }
                    });
            });
            output.to_pyarray_bound(_py)
        }
    };
}

/// Macro for primitives with 8→4 input/output
macro_rules! des_prim_8to4_fn {
    ($name:ident, $prim_fn:ident) => {
        #[pyfunction]
        pub fn $name<'py>(
            _py: Python<'py>,
            p: &ThreadPool,
            data: PyReadonlyArray2<u8>,
        ) -> Bound<'py, PyArray2<u8>> {
            let data_r = data.as_array();
            let (rows, _cols) = data_r.dim();
            let mut output = Array::zeros((rows, 4));
            p.on_worker(_py, || {
                output
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, mut row)| {
                        let mut buf = [0u8; 8];
                        for b in 0..8 {
                            buf[b] = data_r[[i, b]];
                        }
                        let result = $prim_fn(&buf);
                        for b in 0..4 {
                            row[b] = result[b];
                        }
                    });
            });
            output.to_pyarray_bound(_py)
        }
    };
}

/// DES key schedule: (N, 8) keys → (N, 16, 8) round keys (6-bit words).
#[pyfunction]
pub fn des_key_schedule_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    keys: PyReadonlyArray2<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let keys_r = keys.as_array();
    let (rows, _cols) = keys_r.dim();
    let mut output = Array::zeros((rows, 16, 8));
    p.on_worker(_py, || {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut rounds)| {
                let mut key_arr = [0u8; 8];
                for b in 0..8 {
                    key_arr[b] = keys_r[[i, b]];
                }
                let ks = key_schedule(&key_arr);
                for r in 0..16 {
                    for w in 0..8 {
                        rounds[[r, w]] = ks[r][w];
                    }
                }
            });
    });
    output.to_pyarray_bound(_py)
}

des_prim_8to8_fn!(des_initial_permutation_r, initial_permutation);
des_prim_4to8_fn!(des_expansive_permutation_r, expansive_permutation);
des_prim_8to8_fn!(des_sboxes_r, sboxes);
des_prim_8to4_fn!(des_permutation_p_r, permutation_p);
des_prim_4to8_fn!(des_inv_permutation_p_r, inv_permutation_p);
des_prim_8to8_fn!(des_final_permutation_r, final_permutation);
