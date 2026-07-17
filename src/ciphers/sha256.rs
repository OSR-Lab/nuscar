use ndarray::{prelude::*, Zip};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::*;
use super::hash_common::{BLOCK_SIZE, bytes_to_u32, compute_known_sum, hmac_pad_plaintext, parse_iv_8 as parse_iv};

// ============ SHA-256 Constants ============

const HASH_SIZE: usize = 32;

const INITIAL_HASH: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const ROUND_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// ============ SHA-256 Internal Functions ============

#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline]
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

#[inline]
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

#[inline]
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

#[inline]
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// SHA-256 message expansion: 64-byte block -> W[64]
fn sha256_expand_message(block: &[u8]) -> [u32; 64] {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        w[i] = small_sigma1(w[i - 2])
            .wrapping_add(w[i - 7])
            .wrapping_add(small_sigma0(w[i - 15]))
            .wrapping_add(w[i - 16]);
    }
    w
}

/// SHA-256 64-round compression
fn sha256_compress_loop(current_hash: &[u32; 8], w: &[u32; 64]) -> [u32; 8] {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *current_hash;

    for i in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(ROUND_K[i])
            .wrapping_add(w[i]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    [
        current_hash[0].wrapping_add(a),
        current_hash[1].wrapping_add(b),
        current_hash[2].wrapping_add(c),
        current_hash[3].wrapping_add(d),
        current_hash[4].wrapping_add(e),
        current_hash[5].wrapping_add(f),
        current_hash[6].wrapping_add(g),
        current_hash[7].wrapping_add(h),
    ]
}

/// SHA-256 hash with optional custom IV
fn sha256_hash_with_iv(data: &[u8], iv: Option<&[u32; 8]>) -> [u8; 32] {
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
        let w = sha256_expand_message(chunk);
        h = sha256_compress_loop(&h, &w);
    }

    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// Standard SHA-256 hash (no IV)
#[inline]
fn sha256_hash_internal(data: &[u8]) -> [u8; 32] {
    sha256_hash_with_iv(data, None)
}

// ============ PyFunction Exports ============

// --- Core functions ---

#[pyfunction]
#[pyo3(signature = (data, iv=None))]
pub fn sha256_hash_r(data: Vec<u8>, iv: Option<Vec<u32>>) -> Vec<u8> {
    let iv_arr = parse_iv(&iv);
    sha256_hash_with_iv(&data, iv_arr.as_ref()).to_vec()
}

/// Batch SHA-256 hash for 2D array
#[pyfunction]
#[pyo3(signature = (p, data, iv=None))]
pub fn sha256_hash_batch_r<'py>(
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
                let hash = sha256_hash_with_iv(&row_bytes, iv_arr.as_ref());
                for i in 0..HASH_SIZE {
                    out[i] = hash[i];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn sha256_expand_message_r(block: Vec<u8>) -> Vec<u32> {
    let w = sha256_expand_message(&block);
    w.to_vec()
}

#[pyfunction]
pub fn sha256_hmac_r(key: Vec<u8>, msg: Vec<u8>) -> Vec<u8> {
    use super::hash_common::hmac_derive_keys;

    let mut k = if key.len() > BLOCK_SIZE {
        sha256_hash_internal(&key).to_vec()
    } else {
        key
    };
    k.resize(BLOCK_SIZE, 0);
    let (inner_key, outer_key) = hmac_derive_keys(&k);

    let mut inner_data = Vec::from(&inner_key[..]);
    inner_data.extend_from_slice(&msg);
    let inner_hash = sha256_hash_internal(&inner_data);

    let mut outer_data = Vec::from(&outer_key[..]);
    outer_data.extend_from_slice(&inner_hash);
    sha256_hash_internal(&outer_data).to_vec()
}

#[pyfunction]
pub fn sha256_compute_hi_ho_r(padded_key: Vec<u8>) -> Vec<u32> {
    let w = sha256_expand_message(&padded_key);
    let h = sha256_compress_loop(&INITIAL_HASH, &w);
    h.to_vec()
}

// --- Batch compute functions ---

#[pyfunction]
pub fn sha256_compute_w_r<'py>(
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
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
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

// --- Attack functions ---

/// Attack delta0 byte-by-byte. Leakage: t10 = delta0 + W[0]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, byte_index, guesses, known_bytes=None))]
pub fn sha256_attack_delta0_hw_r<'py>(
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
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let w0 = w[0];

                for (j, &guess) in guesses.iter().enumerate() {
                    let t10 = w0
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((t10 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack t20 byte-by-byte. Leakage: a10 = t10 + t20
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, byte_index, guesses, known_bytes=None))]
pub fn sha256_attack_t20_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let t10 = delta0_32.wrapping_add(w[0]);

                for (j, &guess) in guesses.iter().enumerate() {
                    let a10 = t10
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((a10 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack a0 or b0, all 4 bytes at once. Leakage: a1 XOR guess
/// a1 = t10 + t20, output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sha256_attack_a0_or_b0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    t20: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let t20_32 = bytes_to_u32(t20.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let t10 = delta0_32.wrapping_add(w[0]);
                let a1 = t10.wrapping_add(t20_32);
                let a1_bytes = a1.to_be_bytes();

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(a1_bytes[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack d0 byte-by-byte. Leakage: e1 = d0 + t10
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, byte_index, guesses, known_bytes=None))]
pub fn sha256_attack_d0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let t10 = delta0_32.wrapping_add(w[0]);

                for (j, &guess) in guesses.iter().enumerate() {
                    let e1 = t10
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((e1 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack e0, all 4 bytes at once. Leakage: e1 XOR guess, e1 = d0 + t10
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sha256_attack_e0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    d0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let d0_32 = bytes_to_u32(d0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let t10 = delta0_32.wrapping_add(w[0]);
                let e1 = d0_32.wrapping_add(t10);
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

/// Attack f0, all 4 bytes at once. Leakage: ~e1 XOR guess, e1 = d0 + t10
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sha256_attack_f0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    d0: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let d0_32 = bytes_to_u32(d0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);
                let t10 = delta0_32.wrapping_add(w[0]);
                let e1 = d0_32.wrapping_add(t10);
                let not_e1 = !e1;
                let not_e1_bytes = not_e1.to_be_bytes();

                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        out_row[[j, k]] = HW_LUT[(not_e1_bytes[k] ^ guess) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack g0 byte-by-byte. Leakage: t11 = g0 + Sigma1(e1) + Ch(e1,e0,f0) + K[1] + w1
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, d0, e0, f0, byte_index, guesses, known_bytes=None))]
pub fn sha256_attack_g0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    d0: PyReadonlyArray1<u8>,
    e0: PyReadonlyArray1<u8>,
    f0: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let d0_32 = bytes_to_u32(d0.as_slice().unwrap());
    let e0_32 = bytes_to_u32(e0.as_slice().unwrap());
    let f0_32 = bytes_to_u32(f0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);

                let t10 = delta0_32.wrapping_add(w[0]);
                let e1 = d0_32.wrapping_add(t10);

                // delta_g0 = Sigma1(e1) + Ch(e1, e0, f0) + K[1] + w1
                let delta_g0 = big_sigma1(e1)
                    .wrapping_add(ch(e1, e0_32, f0_32))
                    .wrapping_add(ROUND_K[1])
                    .wrapping_add(w[1]);

                for (j, &guess) in guesses.iter().enumerate() {
                    let t11 = delta_g0
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((t11 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack c0 byte-by-byte. Leakage: e2 = c0 + t11
/// where t11 = g0 + Sigma1(e1) + Ch(e1,e0,f0) + K[1] + w1
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, d0, e0, f0, g0, byte_index, guesses, known_bytes=None))]
pub fn sha256_attack_c0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    d0: PyReadonlyArray1<u8>,
    e0: PyReadonlyArray1<u8>,
    f0: PyReadonlyArray1<u8>,
    g0: PyReadonlyArray1<u8>,
    byte_index: usize,
    guesses: Vec<u8>,
    known_bytes: Option<Vec<u8>>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let shift = (3 - byte_index) * 8;
    let known_sum = compute_known_sum(&known_bytes);

    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let d0_32 = bytes_to_u32(d0.as_slice().unwrap());
    let e0_32 = bytes_to_u32(e0.as_slice().unwrap());
    let f0_32 = bytes_to_u32(f0.as_slice().unwrap());
    let g0_32 = bytes_to_u32(g0.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha256_expand_message(&padded[..BLOCK_SIZE]);

                let t10 = delta0_32.wrapping_add(w[0]);
                let e1 = d0_32.wrapping_add(t10);

                // t11 = g0 + Sigma1(e1) + Ch(e1,e0,f0) + K[1] + w1
                let t11 = g0_32
                    .wrapping_add(big_sigma1(e1))
                    .wrapping_add(ch(e1, e0_32, f0_32))
                    .wrapping_add(ROUND_K[1])
                    .wrapping_add(w[1]);

                // e2 = c0 + t11
                for (j, &guess) in guesses.iter().enumerate() {
                    let e2 = t11
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((e2 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}
