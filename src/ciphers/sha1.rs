use ndarray::{prelude::*, Zip};
use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use crate::*;
use super::hash_common::{BLOCK_SIZE, bytes_to_u32, compute_known_sum, hmac_pad_plaintext};

// ============ SHA-1 Constants ============

const HASH_SIZE: usize = 20;

const INITIAL_HASH: [u32; 5] = [
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0,
];

const ROUND_K: [u32; 4] = [
    0x5A827999, // rounds 0-19
    0x6ED9EBA1, // rounds 20-39
    0x8F1BBCDC, // rounds 40-59
    0xCA62C1D6, // rounds 60-79
];

// ============ SHA-1 Internal Functions ============

#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

#[inline]
fn parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline]
fn round_k(j: usize) -> u32 {
    ROUND_K[j / 20]
}

#[inline]
fn round_f(j: usize, b: u32, c: u32, d: u32) -> u32 {
    match j / 20 {
        0 => ch(b, c, d),
        1 => parity(b, c, d),
        2 => maj(b, c, d),
        3 => parity(b, c, d),
        _ => unreachable!(),
    }
}

/// SHA-1 message expansion: 64-byte block -> W[80]
fn sha1_expand_message(block: &[u8]) -> [u32; 80] {
    let mut w = [0u32; 80];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..80 {
        w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
    }
    w
}

/// SHA-1 80-round compression
fn sha1_compress_loop(current_hash: &[u32; 5], w: &[u32; 80]) -> [u32; 5] {
    let [mut a, mut b, mut c, mut d, mut e] = *current_hash;

    for i in 0..80 {
        let temp = a.rotate_left(5)
            .wrapping_add(round_f(i, b, c, d))
            .wrapping_add(e)
            .wrapping_add(round_k(i))
            .wrapping_add(w[i]);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = temp;
    }

    [
        current_hash[0].wrapping_add(a),
        current_hash[1].wrapping_add(b),
        current_hash[2].wrapping_add(c),
        current_hash[3].wrapping_add(d),
        current_hash[4].wrapping_add(e),
    ]
}

/// SHA-1 hash with optional custom IV
fn sha1_hash_with_iv(data: &[u8], iv: Option<&[u32; 5]>) -> [u8; 20] {
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
        let w = sha1_expand_message(chunk);
        h = sha1_compress_loop(&h, &w);
    }

    let mut result = [0u8; 20];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// Standard SHA-1 hash (no IV)
#[inline]
fn sha1_hash_internal(data: &[u8]) -> [u8; 20] {
    sha1_hash_with_iv(data, None)
}

/// Parse optional IV from Python list to [u32; 5]
#[inline]
fn parse_iv(iv: &Option<Vec<u32>>) -> Option<[u32; 5]> {
    iv.as_ref().map(|v| {
        let mut arr = [0u32; 5];
        arr.copy_from_slice(v);
        arr
    })
}

// ============ PyFunction Exports ============

// --- Core functions ---

#[pyfunction]
#[pyo3(signature = (data, iv=None))]
pub fn sha1_hash_r(data: Vec<u8>, iv: Option<Vec<u32>>) -> Vec<u8> {
    let iv_arr = parse_iv(&iv);
    sha1_hash_with_iv(&data, iv_arr.as_ref()).to_vec()
}

/// Batch SHA-1 hash for 2D array
#[pyfunction]
#[pyo3(signature = (p, data, iv=None))]
pub fn sha1_hash_batch_r<'py>(
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
                let hash = sha1_hash_with_iv(&row_bytes, iv_arr.as_ref());
                for i in 0..HASH_SIZE {
                    out[i] = hash[i];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

#[pyfunction]
pub fn sha1_expand_message_r(block: Vec<u8>) -> Vec<u32> {
    let w = sha1_expand_message(&block);
    w.to_vec()
}

#[pyfunction]
pub fn sha1_hmac_r(key: Vec<u8>, msg: Vec<u8>) -> Vec<u8> {
    use super::hash_common::hmac_derive_keys;

    let mut k = if key.len() > BLOCK_SIZE {
        sha1_hash_internal(&key).to_vec()
    } else {
        key
    };
    k.resize(BLOCK_SIZE, 0);
    let (inner_key, outer_key) = hmac_derive_keys(&k);

    let mut inner_data = Vec::from(&inner_key[..]);
    inner_data.extend_from_slice(&msg);
    let inner_hash = sha1_hash_internal(&inner_data);

    let mut outer_data = Vec::from(&outer_key[..]);
    outer_data.extend_from_slice(&inner_hash);
    sha1_hash_internal(&outer_data).to_vec()
}

#[pyfunction]
pub fn sha1_compute_hi_ho_r(padded_key: Vec<u8>) -> Vec<u32> {
    let w = sha1_expand_message(&padded_key);
    let h = sha1_compress_loop(&INITIAL_HASH, &w);
    h.to_vec()
}

// --- Batch compute functions ---

#[pyfunction]
pub fn sha1_compute_w_r<'py>(
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
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);
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

/// Attack delta0 byte-by-byte. Leakage: T0 = delta0 + W[0]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, byte_index, guesses, known_bytes=None))]
pub fn sha1_attack_delta0_hw_r<'py>(
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
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);
                let w0 = w[0];

                for (j, &guess) in guesses.iter().enumerate() {
                    let t0 = w0
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((t0 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack R1 byte-by-byte. Leakage: T1 = ROTL(T0,5) + R1 + W[1]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, byte_index, guesses, known_bytes=None))]
pub fn sha1_attack_r1_hw_r<'py>(
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
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);

                // T0 = delta0 + W[0]
                let t0 = delta0_32.wrapping_add(w[0]);
                // known_part = ROTL(T0,5) + W[1]
                let known_part = t0.rotate_left(5).wrapping_add(w[1]);

                // T1 = known_part + R1
                for (j, &guess) in guesses.iter().enumerate() {
                    let t1 = known_part
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((t1 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack ROTL(a0,30), all 4 bytes. Leakage: Ch(T1, ROTL(T0,30), ROTL(a0,30))
/// Non-linear: F3 = (T1 & ROTL(T0,30)) ^ (~T1 & guess)
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sha1_attack_a0_rot30_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    r1: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let r1_32 = bytes_to_u32(r1.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);

                let t0 = delta0_32.wrapping_add(w[0]);
                let t1 = t0.rotate_left(5).wrapping_add(r1_32).wrapping_add(w[1]);
                let t0_rot30 = t0.rotate_left(30);

                let t1_bytes = t1.to_be_bytes();
                let t0_rot30_bytes = t0_rot30.to_be_bytes();

                // Ch(T1, ROTL(T0,30), guess) = (T1 & ROTL(T0,30)) ^ (~T1 & guess)
                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        let term1 = t1_bytes[k] & t0_rot30_bytes[k];
                        let term2 = (!t1_bytes[k]) & guess;
                        out_row[[j, k]] = HW_LUT[(term1 ^ term2) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack ROTL(b0,30), all 4 bytes. Leakage: Ch(T0, ROTL(a0,30), ROTL(b0,30))
/// Non-linear: F2 = (T0 & ROTL(a0,30)) ^ (~T0 & guess)
/// Output shape: (N, num_guesses, 4)
#[pyfunction]
pub fn sha1_attack_b0_rot30_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    a0_rot30: PyReadonlyArray1<u8>,
    guesses: Vec<u8>,
) -> Bound<'py, PyArray3<u8>> {
    let plain_r = plaintext.as_array();
    let (rows, _cols) = plain_r.dim();
    let num_guesses = guesses.len();
    let delta0_32 = bytes_to_u32(delta0.as_slice().unwrap());
    let a0_rot30_bytes: [u8; 4] = a0_rot30.as_slice().unwrap().try_into().unwrap();

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 4));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);

                let t0 = delta0_32.wrapping_add(w[0]);
                let t0_bytes = t0.to_be_bytes();

                // Ch(T0, ROTL(a0,30), guess) = (T0 & ROTL(a0,30)) ^ (~T0 & guess)
                for (j, &guess) in guesses.iter().enumerate() {
                    for k in 0..4 {
                        let term1 = t0_bytes[k] & a0_rot30_bytes[k];
                        let term2 = (!t0_bytes[k]) & guess;
                        out_row[[j, k]] = HW_LUT[(term1 ^ term2) as usize];
                    }
                }
            });
    });
    res.to_pyarray_bound(_py)
}

/// Attack c0 byte-by-byte. Leakage: T2 = ROTL(T1,5) + Ch(T0,a0_rot30,b0_rot30) + c0 + K[2] + W[2]
/// Output shape: (N, num_guesses, 1)
#[pyfunction(signature = (p, plaintext, delta0, r1, a0_rot30, b0_rot30, byte_index, guesses, known_bytes=None))]
pub fn sha1_attack_c0_hw_r<'py>(
    _py: Python<'py>,
    p: &ThreadPool,
    plaintext: PyReadonlyArray2<u8>,
    delta0: PyReadonlyArray1<u8>,
    r1: PyReadonlyArray1<u8>,
    a0_rot30: PyReadonlyArray1<u8>,
    b0_rot30: PyReadonlyArray1<u8>,
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
    let r1_32 = bytes_to_u32(r1.as_slice().unwrap());
    let a0_rot30_32 = bytes_to_u32(a0_rot30.as_slice().unwrap());
    let b0_rot30_32 = bytes_to_u32(b0_rot30.as_slice().unwrap());

    let mut res: Array3<u8> = Array::zeros((rows, num_guesses, 1));

    p.on_worker(_py, || {
        Zip::from(res.axis_iter_mut(Axis(0)))
            .and(plain_r.axis_iter(Axis(0)))
            .par_for_each(|mut out_row, row| {
                let row_bytes: Vec<u8> = row.iter().copied().collect();
                let padded = hmac_pad_plaintext(&row_bytes);
                let w = sha1_expand_message(&padded[..BLOCK_SIZE]);

                // T0 = delta0 + W[0]
                let t0 = delta0_32.wrapping_add(w[0]);
                // T1 = ROTL(T0,5) + R1 + W[1]
                let t1 = t0.rotate_left(5).wrapping_add(r1_32).wrapping_add(w[1]);

                // Ch(T0, a0_rot30, b0_rot30)
                let ch_val = ch(t0, a0_rot30_32, b0_rot30_32);

                // known_part = ROTL(T1,5) + Ch(...) + K[2] + W[2]
                let known_part = t1.rotate_left(5)
                    .wrapping_add(ch_val)
                    .wrapping_add(round_k(2))
                    .wrapping_add(w[2]);

                // T2 = known_part + c0
                for (j, &guess) in guesses.iter().enumerate() {
                    let t2 = known_part
                        .wrapping_add((guess as u32) << shift)
                        .wrapping_add(known_sum);
                    let target_byte = ((t2 >> shift) & 0xFF) as u8;
                    out_row[[j, 0]] = HW_LUT[target_byte as usize];
                }
            });
    });
    res.to_pyarray_bound(_py)
}
