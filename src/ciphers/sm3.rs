use ndarray::{prelude::*, Zip};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::*;
use super::hash_common::{BLOCK_SIZE, bytes_to_u32, compute_known_sum, hmac_pad_plaintext, parse_iv_8 as parse_iv};

// ============ SM3 Constants ============

const HASH_SIZE: usize = 32;

const INITIAL_HASH: [u32; 8] = [
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e,
];

// ============ SM3 Internal Functions ============

#[inline]
fn p0(x: u32) -> u32 {
    x ^ x.rotate_left(9) ^ x.rotate_left(17)
}

#[inline]
fn p1(x: u32) -> u32 {
    x ^ x.rotate_left(15) ^ x.rotate_left(23)
}

#[inline]
fn round_t(j: usize) -> u32 {
    if j < 16 { 0x79cc4519 } else { 0x7a879d8a }
}

/// SM3 message expansion: 64-byte block -> (W[68], W'[64])
fn sm3_expand_message(block: &[u8]) -> ([u32; 68], [u32; 64]) {
    let mut w = [0u32; 68];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for j in 16..68 {
        let wj = w[j - 16] ^ w[j - 9] ^ w[j - 3].rotate_left(15);
        w[j] = p1(wj) ^ w[j - 13].rotate_left(7) ^ w[j - 6];
    }
    let mut w_prime = [0u32; 64];
    for j in 0..64 {
        w_prime[j] = w[j] ^ w[j + 4];
    }
    (w, w_prime)
}

/// SM3 64-round compression
fn sm3_compress_loop(current_hash: &[u32; 8], w: &[u32; 68], w_prime: &[u32; 64]) -> [u32; 8] {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *current_hash;

    for j in 0..64 {
        let ss1 = (a.rotate_left(12)
            .wrapping_add(e)
            .wrapping_add(round_t(j).rotate_left((j as u32) & 31)))
        .rotate_left(7);
        let ss2 = ss1 ^ a.rotate_left(12);

        let (tt1, tt2) = if j < 16 {
            (
                (a ^ b ^ c).wrapping_add(d).wrapping_add(ss2).wrapping_add(w_prime[j]),
                (e ^ f ^ g).wrapping_add(h).wrapping_add(ss1).wrapping_add(w[j]),
            )
        } else {
            (
                ((a & b) | (a & c) | (b & c))
                    .wrapping_add(d)
                    .wrapping_add(ss2)
                    .wrapping_add(w_prime[j]),
                ((e & f) | (!e & g))
                    .wrapping_add(h)
                    .wrapping_add(ss1)
                    .wrapping_add(w[j]),
            )
        };

        d = c;
        c = b.rotate_left(9);
        b = a;
        a = tt1;
        h = g;
        g = f.rotate_left(19);
        f = e;
        e = p0(tt2);
    }

    [
        current_hash[0] ^ a,
        current_hash[1] ^ b,
        current_hash[2] ^ c,
        current_hash[3] ^ d,
        current_hash[4] ^ e,
        current_hash[5] ^ f,
        current_hash[6] ^ g,
        current_hash[7] ^ h,
    ]
}

/// SM3 hash with optional custom IV (for HMAC second-block scenarios)
/// When iv is Some, length_in_bits = (BLOCK_SIZE + data.len()) * 8
/// When iv is None, standard SM3: length_in_bits = data.len() * 8
fn sm3_hash_with_iv(data: &[u8], iv: Option<&[u32; 8]>) -> [u8; 32] {
    let length_in_bits = if iv.is_some() {
        ((BLOCK_SIZE + data.len()) * 8) as u64
    } else {
        (data.len() * 8) as u64
    };
    let mut padded = Vec::from(data);
    padded.push(0x80);
    while (padded.len() * 8) % 512 != 448 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&length_in_bits.to_be_bytes());

    let mut h = match iv {
        Some(custom_iv) => *custom_iv,
        None => INITIAL_HASH,
    };
    for chunk in padded.chunks(BLOCK_SIZE) {
        let (w, w_prime) = sm3_expand_message(chunk);
        h = sm3_compress_loop(&h, &w, &w_prime);
    }

    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// Standard SM3 hash (no IV)
#[inline]
fn sm3_hash_internal(data: &[u8]) -> [u8; 32] {
    sm3_hash_with_iv(data, None)
}

// ============ Shared computation for round 1 attacks ============

/// Compute values needed for round 1 attacks (c0, g0).
/// Returns (A, E, SS1_1, SS2_1)
#[inline]
fn compute_round1_values(
    w: &[u32; 68],
    w_prime: &[u32; 64],
    delta1_0_32: u32,
    delta2_0_32: u32,
) -> (u32, u32, u32, u32) {
    // A = TT1_0 = delta1_0 + W'[0]
    let a_val = delta1_0_32.wrapping_add(w_prime[0]);
    // E = P0(TT2_0) = P0(delta2_0 + W[0])
    let tt2_0 = delta2_0_32.wrapping_add(w[0]);
    let e_val = p0(tt2_0);
    // SS1_1 = rotl((rotl(A,12) + E + rotl(T[1],1)) & mask, 7)
    let t1 = round_t(1);
    let ss1_1 = (a_val.rotate_left(12)
        .wrapping_add(e_val)
        .wrapping_add(t1.rotate_left(1)))
    .rotate_left(7);
    // SS2_1 = SS1_1 ^ rotl(A, 12)
    let ss2_1 = ss1_1 ^ a_val.rotate_left(12);
    (a_val, e_val, ss1_1, ss2_1)
}

// ============ PyFunction Exports ============

// --- Core functions ---

#[pyfunction]
#[pyo3(signature = (data, iv=None))]
pub fn sm3_hash_r(data: Vec<u8>, iv: Option<Vec<u32>>) -> Vec<u8> {
    let iv_arr = parse_iv(&iv);
    sm3_hash_with_iv(&data, iv_arr.as_ref()).to_vec()
}

/// Batch SM3 hash for 2D array, with optional custom IV and rayon parallelism
#[pyfunction]
#[pyo3(signature = (p, data, iv=None))]
pub fn sm3_hash_batch_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    data: PyReadonlyArray2<u8>,
    iv: Option<Vec<u32>>,
) -> Bound<'py, PyArray2<u8>> {
    let data_r = data.as_array();
    let (rows, _cols) = data_r.dim();
    let iv_arr = parse_iv(&iv);

    let mut res: Array2<u8> = Array::zeros((rows, HASH_SIZE));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(data_r.axis_iter(Axis(0)))
            .par_for_each(|mut out, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let hash = sm3_hash_with_iv(&row_bytes, iv_arr.as_ref());
                for i in 0..HASH_SIZE {
                    out[i] = hash[i];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn sm3_expand_message_r(block: Vec<u8>) -> (Vec<u32>, Vec<u32>) {
    let (w, w_prime) = sm3_expand_message(&block);
    (w.to_vec(), w_prime.to_vec())
}

#[pyfunction]
pub fn sm3_hmac_r(key: Vec<u8>, msg: Vec<u8>) -> Vec<u8> {
    use super::hash_common::hmac_derive_keys;

    let mut k = if key.len() > BLOCK_SIZE {
        sm3_hash_internal(&key).to_vec()
    } else {
        key
    };
    k.resize(BLOCK_SIZE, 0);
    let (inner_key, outer_key) = hmac_derive_keys(&k);

    let mut inner_data = Vec::from(&inner_key[..]);
    inner_data.extend_from_slice(&msg);
    let inner_hash = sm3_hash_internal(&inner_data);

    let mut outer_data = Vec::from(&outer_key[..]);
    outer_data.extend_from_slice(&inner_hash);
    sm3_hash_internal(&outer_data).to_vec()
}

#[pyfunction]
pub fn sm3_compute_hi_ho_r(padded_key: Vec<u8>) -> Vec<u32> {
    let (w, w_prime) = sm3_expand_message(&padded_key);
    let h = sm3_compress_loop(&INITIAL_HASH, &w, &w_prime);
    h.to_vec()
}

// --- Batch compute functions ---

#[pyfunction]
pub fn sm3_compute_w_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    w_index: usize,
) -> Bound<'py, PyArray2<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let mut res: Array2<u8> = Array::zeros((rows, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, _) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let val = w[w_index];
                let bytes = val.to_be_bytes();
                out[0] = bytes[0];
                out[1] = bytes[1];
                out[2] = bytes[2];
                out[3] = bytes[3];
            });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn sm3_compute_w_prime_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    w_index: usize,
) -> Bound<'py, PyArray2<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let mut res: Array2<u8> = Array::zeros((rows, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (_, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let val = w_prime[w_index];
                let bytes = val.to_be_bytes();
                out[0] = bytes[0];
                out[1] = bytes[1];
                out[2] = bytes[2];
                out[3] = bytes[3];
            });
    });
    res.to_pyarray_bound(_py)
}

// --- Attack functions ---

/// Step 1: attack delta1_0, byte-by-byte. Leakage: TT1_0 = delta1_0 + W'[0]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, byte_index, guesses, known_bytes=None))]
pub fn sm3_attack_delta1_0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (_, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let w_prime0 = w_prime[0];

                for (j, &guess) in guesses.iter().enumerate() {
                    let tt1_0 = w_prime0
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((tt1_0 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 2: attack delta2_0, byte-by-byte. Leakage: TT2_0 = delta2_0 + W[0]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, byte_index, guesses, known_bytes=None))]
pub fn sm3_attack_delta2_0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, _) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let w0 = w[0];

                for (j, &guess) in guesses.iter().enumerate() {
                    let tt2_0 = w0
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((tt2_0 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 3: attack a0, all 4 bytes at once. Leakage: TT1_0 ^ a0
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sm3_attack_a0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta1_0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta1_0_slice = delta1_0.as_slice().unwrap();
    let delta1_0_32 = bytes_to_u32(delta1_0_slice);

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (_, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let tt1_0 = delta1_0_32.wrapping_add(w_prime[0]);
                let tt1_0_bytes = tt1_0.to_be_bytes();

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(tt1_0_bytes[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 4: attack rotl9(b0), all 4 bytes at once. Leakage: TT1_0 ^ a0 ^ rotl9(b0)
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sm3_attack_b0_rotl9_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta1_0: PyReadonlyArray1<u8>,
    a0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta1_0_32 = bytes_to_u32(delta1_0.as_slice().unwrap());
    let a0_bytes: [u8; 4] = a0.as_slice().unwrap().try_into().unwrap();

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (_, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let tt1_0 = delta1_0_32.wrapping_add(w_prime[0]);
                let tt1_0_bytes = tt1_0.to_be_bytes();

                // tt1_xor_a0
                let xored = [
                    tt1_0_bytes[0] ^ a0_bytes[0],
                    tt1_0_bytes[1] ^ a0_bytes[1],
                    tt1_0_bytes[2] ^ a0_bytes[2],
                    tt1_0_bytes[3] ^ a0_bytes[3],
                ];

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(xored[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 5: attack e0, all 4 bytes at once. Leakage: P0(TT2_0) ^ e0
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sm3_attack_e0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta2_0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta2_0_32 = bytes_to_u32(delta2_0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, _) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let tt2_0 = delta2_0_32.wrapping_add(w[0]);
                let e1 = p0(tt2_0);
                let e1_bytes = e1.to_be_bytes();

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(e1_bytes[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 6: attack rotl19(f0), all 4 bytes at once. Leakage: P0(TT2_0) ^ e0 ^ rotl19(f0)
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sm3_attack_f0_rotl19_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta2_0: PyReadonlyArray1<u8>,
    e0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta2_0_32 = bytes_to_u32(delta2_0.as_slice().unwrap());
    let e0_bytes: [u8; 4] = e0.as_slice().unwrap().try_into().unwrap();

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, _) = sm3_expand_message(&padded[..BLOCK_SIZE]);
                let tt2_0 = delta2_0_32.wrapping_add(w[0]);
                let e1 = p0(tt2_0);
                let e1_bytes = e1.to_be_bytes();

                let xored = [
                    e1_bytes[0] ^ e0_bytes[0],
                    e1_bytes[1] ^ e0_bytes[1],
                    e1_bytes[2] ^ e0_bytes[2],
                    e1_bytes[3] ^ e0_bytes[3],
                ];

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(xored[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 7: attack c0, byte-by-byte. Leakage: TT1_1 = delta1_1 + C0 + W'[1]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta1_0, delta2_0, a0, b0_rotl9, byte_index, guesses, known_bytes=None))]
pub fn sm3_attack_c0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta1_0: PyReadonlyArray1<u8>,
    delta2_0: PyReadonlyArray1<u8>,
    a0: PyReadonlyArray1<u8>,
    b0_rotl9: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let delta1_0_32 = bytes_to_u32(delta1_0.as_slice().unwrap());
    let delta2_0_32 = bytes_to_u32(delta2_0.as_slice().unwrap());
    let a0_32 = bytes_to_u32(a0.as_slice().unwrap());
    let b0_rotl9_32 = bytes_to_u32(b0_rotl9.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);

                // Compute round 1 values (single expansion, no redundancy)
                let (a_val, _e_val, _ss1_1, ss2_1) =
                    compute_round1_values(&w, &w_prime, delta1_0_32, delta2_0_32);

                // delta1_1 = xor3(A, a0, rotl9(b0)) + SS2_1
                let xor_abc = a_val ^ a0_32 ^ b0_rotl9_32;
                let delta1_1 = xor_abc.wrapping_add(ss2_1);

                // TT1_1 = delta1_1 + W'[1] + C0_guess + known_sum
                let base = delta1_1.wrapping_add(w_prime[1]);

                for (j, &guess) in guesses.iter().enumerate() {
                    let tt1_1 = base
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((tt1_1 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Step 8: attack g0, byte-by-byte. Leakage: TT2_1 = delta2_1 + G0 + W[1]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta1_0, delta2_0, e0, f0_rotl19, byte_index, guesses, known_bytes=None))]
pub fn sm3_attack_g0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta1_0: PyReadonlyArray1<u8>,
    delta2_0: PyReadonlyArray1<u8>,
    e0: PyReadonlyArray1<u8>,
    f0_rotl19: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let delta1_0_32 = bytes_to_u32(delta1_0.as_slice().unwrap());
    let delta2_0_32 = bytes_to_u32(delta2_0.as_slice().unwrap());
    let e0_32 = bytes_to_u32(e0.as_slice().unwrap());
    let f0_rotl19_32 = bytes_to_u32(f0_rotl19.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let (w, w_prime) = sm3_expand_message(&padded[..BLOCK_SIZE]);

                // Compute round 1 values
                let (_a_val, e_val, ss1_1, _ss2_1) =
                    compute_round1_values(&w, &w_prime, delta1_0_32, delta2_0_32);

                // delta2_1 = xor3(E, e0, rotl19(f0)) + SS1_1
                let xor_efg = e_val ^ e0_32 ^ f0_rotl19_32;
                let delta2_1 = xor_efg.wrapping_add(ss1_1);

                // TT2_1 = delta2_1 + W[1] + G0_guess + known_sum
                let base = delta2_1.wrapping_add(w[1]);

                for (j, &guess) in guesses.iter().enumerate() {
                    let tt2_1 = base
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((tt2_1 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}
