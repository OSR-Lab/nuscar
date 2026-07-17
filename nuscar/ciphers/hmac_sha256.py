# Copyright(c)  2026. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as np
import nuscar
import nuscar.nuscar_rust as _n_rust

BLOCK_SIZE = 64

# Initial hash values (H_0 to H_7), first 32 bits of fractional parts of square roots of first 8 primes
_INITIAL_HASH = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]

# Round constants (K_0 to K_63), first 32 bits of fractional parts of cube roots of first 64 primes
_ROUND_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]


# Bitwise operation helper functions
def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF


def shr(x, n):
    return x >> n


# Logic functions
def ch(x, y, z):
    return (x & y) ^ (~x & z)


def maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def big_sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def big_sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def small_sigma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def small_sigma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)


def uint32_to_uint8_array(w: np.ndarray | int) -> np.ndarray:
    """
    Convert a uint32 integer array to a uint8 byte array (big-endian).
    Args:
        w: numpy array of uint32, shape (n,) or scalar
    Returns:
        numpy array of uint8
        - If input is scalar or shape (), returns shape (4,)
        - If input shape is (n,), returns shape (n, 4)
    """
    w = np.asarray(w, dtype=np.uint32)
    if w.ndim == 0:
        return np.array([(w >> 24) & 0xFF, (w >> 16) & 0xFF, (w >> 8) & 0xFF, w & 0xFF], dtype=np.uint8)
    else:
        result = np.zeros((w.shape[0], 4), dtype=np.uint8)
        result[:, 0] = (w >> 24) & 0xFF
        result[:, 1] = (w >> 16) & 0xFF
        result[:, 2] = (w >> 8) & 0xFF
        result[:, 3] = w & 0xFF
        return result

def uint8_array_to_uint32(arr: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 byte array to a uint32 integer array (big-endian).
    Args:
        arr: numpy array of uint8, shape (4,) or (n, 4)
    Returns:
        numpy array of uint32
        - If input shape is (4,), returns scalar (0-dim array)
        - If input shape is (n, 4), returns shape (n,)
    """
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 1:
        if arr.shape[0] != 4:
            raise ValueError("1D input must be an array of length 4")
        return ((np.uint32(arr[0]) << 24) |
                (np.uint32(arr[1]) << 16) |
                (np.uint32(arr[2]) << 8)  |
                np.uint32(arr[3]))
    else:
        if arr.shape[1] != 4:
            raise ValueError("Second dimension of 2D input must be 4")
        return ((arr[:, 0].astype(np.uint32) << 24) |
                (arr[:, 1].astype(np.uint32) << 16) |
                (arr[:, 2].astype(np.uint32) << 8)  |
                (arr[:, 3].astype(np.uint32)))


def compute_w(plaintext: np.ndarray, w_index: int = 0) -> np.ndarray:
    """
    Compute the w_index-th message word of HMAC-SHA256 inner hash.
    Args:
        plaintext: 2D numpy uint8 array of plaintext data, shape (n, m), m is byte length
        w_index: message word index, default 0 (w0), optionally 1 (w1), etc.

    Returns:
        numpy uint8 array containing w values for each plaintext, shape (n, 4)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    plaintext = np.ascontiguousarray(plaintext, dtype=np.uint8)
    return _n_rust.sha256_compute_w_r(nuscar._global_pool, plaintext, w_index)


def compute_w0(plaintext: np.ndarray) -> np.ndarray:
    # Compute the first message word w0 of HMAC-SHA256 inner hash.
    return compute_w(plaintext, 0)


def compute_w1(plaintext: np.ndarray) -> np.ndarray:
    # Compute the second message word w1 of HMAC-SHA256 inner hash.
    return compute_w(plaintext, 1)


def compute_t10(plaintext: np.ndarray, delta0: np.ndarray):
    """
    Compute the t1 value in the first round compression of the second block of HMAC-SHA256.
    t1 = h + Sigma1(e) + Ch(e,f,g) + K[0] + W[0]
       = delta0 + w0
    Args:
        plaintext: 2D numpy uint8 array, shape (n, m), m is byte length
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
    Returns:
        t1 value as 32-bit integer, shape (n,) or scalar
    """
    w0 = compute_w0(plaintext)
    w0_32bit = uint8_array_to_uint32(w0)
    delta0_32bit = uint8_array_to_uint32(delta0) if delta0.ndim == 2 else uint8_array_to_uint32(delta0[np.newaxis, :])
    return (w0_32bit + delta0_32bit) & 0xFFFFFFFF


def compute_t10_hw(plaintext: np.ndarray, delta0: np.ndarray):
    res = uint32_to_uint8_array(compute_t10(plaintext, delta0))
    return nuscar.leakmodel.leakage_model_hw(res, nb_words=1)


def compute_padded_key(key: bytes, pad: int) -> bytes:
    """
    Compute the HMAC padded key.
    Args:
        key: original key, bytes type
        pad: padding byte value, 0x36 (ipad) or 0x5c (opad)
    Returns:
        padded and XORed key, length BLOCK_SIZE (64 bytes)
    """
    # If key length exceeds BLOCK_SIZE, hash it first
    if len(key) > BLOCK_SIZE:
        key = sha256(key)

    # Pad to BLOCK_SIZE
    if len(key) < BLOCK_SIZE:
        key = key + b'\x00' * (BLOCK_SIZE - len(key))

    # XOR with pad
    pad_bytes = bytes([pad] * BLOCK_SIZE)
    padded_key = bytes(k ^ p for k, p in zip(key, pad_bytes))
    return padded_key


def compute_hi_ho(padded_key: bytes) -> list:
    # Compute the initial hash values of HMAC-SHA256 inner/outer hash (after pad processing)
    return list(_n_rust.sha256_compute_hi_ho_r(list(padded_key)))


def compute_delta0(padded_key: bytes) -> np.ndarray:
    """
    Compute the delta0 value of HMAC-SHA256 (h + Sigma1(e) + Ch(e,f,g) + K[0]).
    delta0 is the partial intermediate value of the HMAC-SHA256 inner hash when entering
    the second block (containing plaintext) after processing ipad, during the first round
    of compression. It does not depend on plaintext, only on the key.

    Args:
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)

    Returns:
        4-byte representation of delta0, numpy uint8 array, shape (4,), big-endian
    """
    a, b, c, d, e, f_val, g, h_val = compute_hi_ho(padded_key)
    delta0 = (h_val + big_sigma1(e) + ch(e, f_val, g) + _ROUND_K[0]) & 0xFFFFFFFF
    return uint32_to_uint8_array(delta0)


def compute_t20(padded_key: bytes) -> np.ndarray:
    """
    Compute the t2 value of the first round of the second block of HMAC-SHA256.

    Args:
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)

    Returns:
        4-byte representation of t20, numpy uint8 array, shape (4,)
    """
    a, b, c, d, e, f_val, g, h_val = compute_hi_ho(padded_key)
    t20 = (big_sigma0(a) + maj(a, b, c)) & 0xFFFFFFFF
    return uint32_to_uint8_array(t20)


def attack_delta0_hw(plaintext: np.ndarray, byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Generic delta0 byte attack function.
    Args:
        plaintext: array of shape (N, ...), first 4 bytes are w0
        byte_index: byte index to attack (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB, e.g. [byte3, byte2, byte1]
        guesses: guess value array, default 0-255

    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_delta0_hw_r(
        nuscar._global_pool, plaintext, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )



def compute_a10(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray):
    """
    Compute the a value after the first round compression of the second block of HMAC-SHA256.
    In the SHA-256 compression function, the new a value is computed as:
    a = t1 + t2
    where:
    - t1 = h + Sigma1(e) + Ch(e,f,g) + K[0] + W[0]
    - t2 = Sigma0(a) + Maj(a,b,c)
    Args:
        plaintext: 2D numpy uint8 array, shape (n, m), m is byte length
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        t20: t2 value, numpy uint8 array, shape (4,)
    Returns:
        a1 value as 32-bit integer, shape (n,) or scalar
    """
    t10 = compute_t10(plaintext, delta0)
    return (t10 + uint8_array_to_uint32(t20)) & 0xFFFFFFFF


def compute_a10_hw(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray):
    res = compute_a10(plaintext, delta0, t20)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_t20_hw(plaintext: np.ndarray, delta0: np.ndarray,
                  byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Generic t20 byte attack function. Compute the Hamming weight of the target byte
    of a10 = t10 + t20 using known t10 and guessed t20 bytes.
    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        byte_index: byte index to attack (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB, e.g. [byte3, byte2, byte1]
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_t20_hw_r(
        nuscar._global_pool, plaintext, delta0, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def compute_a0_to_h0(padded_key: bytes) -> dict:
    """
    Compute all state values a0 to h0 after compressing the first block (ipad) of HMAC-SHA256.

    Args:
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        dict containing a0 to h0, each value is a numpy uint8 array, shape (4,), big-endian
    """
    a0, b0, c0, d0, e0, f0, g0, h0 = compute_hi_ho(padded_key)
    return {'a0': uint32_to_uint8_array(a0), 'b0': uint32_to_uint8_array(b0), 'c0': uint32_to_uint8_array(c0),
            'd0': uint32_to_uint8_array(d0), 'e0': uint32_to_uint8_array(e0), 'f0': uint32_to_uint8_array(f0),
            'g0': uint32_to_uint8_array(g0), 'h0': uint32_to_uint8_array(h0)}


def compute_a1_and_a0(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray, padded_key: bytes):
    """
    Compute the XOR of a1 (after first round compression) and initial a0.
    a1 is the a value after the first round compression of the second block,
    a0 is the a value after compressing the first block (ipad).
    a1 XOR a0 can be used to recover a0 from known a1.

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        t20: t2 value, numpy uint8 array, shape (4,)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        a1 XOR a0 as 32-bit integer, shape (n,) or scalar
    """
    a0, b, c, d, e, f_val, g, h_val = compute_hi_ho(padded_key)
    a1 = compute_a10(plaintext, delta0, t20)
    return a1 ^ a0


def compute_a1_and_a0_hw(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray, key: bytes):
    res = compute_a1_and_a0(plaintext, delta0, t20, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_a0_or_b0_hw(plaintext, delta0, t20, guesses=np.arange(256, dtype=np.uint8)):
    """
    Attack all 4 bytes of a0/b0 at once.
    Compute the Hamming weight of (a1 XOR a0/b0) in Maj(), where a1 = t10 + t20.

    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        t20: t2 value, numpy uint8 array, shape (4,)
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 4),
        corresponding to byte0, byte1, byte2, byte3 of a0
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_a0_or_b0_hw_r(
        nuscar._global_pool, plaintext, delta0, t20, list(guesses)
    )


def compute_a1_and_b0(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray, padded_key: bytes):
    """
    Compute the XOR of a1 (after first round compression) and initial b0.

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        t20: t2 value, numpy uint8 array, shape (4,)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        a1 XOR b0 as 32-bit integer, shape (n,) or scalar
    """
    a0, b0, c, d, e, f_val, g, h_val = compute_hi_ho(padded_key)
    a1 = compute_a10(plaintext, delta0, t20)
    return a1 ^ b0


def compute_a1_and_b0_hw(plaintext: np.ndarray, delta0: np.ndarray, t20: np.ndarray, key: bytes):
    res = compute_a1_and_b0(plaintext, delta0, t20, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def compute_e1(plaintext: np.ndarray, delta0: np.ndarray, padded_key: bytes):
    """
    Compute the e1 value after the first round compression.

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        e1 as 32-bit integer, shape (n,) or scalar
    """
    a0, b0, c, d0, e, f_val, g, h_val = compute_hi_ho(padded_key)
    t1 = compute_t10(plaintext, delta0)
    return (d0 + t1) & 0xFFFFFFFF


def compute_e1_hw(plaintext: np.ndarray, delta0: np.ndarray, key: bytes):
    res = compute_e1(plaintext, delta0, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_d0_hw(plaintext: np.ndarray, delta0: np.ndarray,
                 byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Generic d0 byte attack function. Compute the Hamming weight of the target byte
    of e1 = d0 + t10 using known t10 and guessed d0 bytes.

    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        byte_index: byte index to attack (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB, e.g. [byte3, byte2, byte1]
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_d0_hw_r(
        nuscar._global_pool, plaintext, delta0, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def compute_e1_and_e0(plaintext: np.ndarray, delta0: np.ndarray, padded_key: bytes):
    """
    Compute the XOR of e1 (after first round compression) and initial e0.

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        e1 XOR e0 as 32-bit integer, shape (n,) or scalar
    """
    a0, b0, c, d0, e0, f_val, g, h_val = compute_hi_ho(padded_key)
    t1 = compute_t10(plaintext, delta0)
    e1 = (d0 + t1) & 0xFFFFFFFF
    return e1 ^ e0


def compute_e1_and_e0_hw(plaintext: np.ndarray, delta0: np.ndarray, key: bytes):
    res = compute_e1_and_e0(plaintext, delta0, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_e0_hw(plaintext: np.ndarray, delta0: np.ndarray, d0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Attack all 4 bytes of e0 at once.
    Compute the Hamming weight of (e1 XOR e0), where e1 = d0 + t10.

    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        d0: d0 value, numpy uint8 array, shape (4,)
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 4), attacking all 4 bytes simultaneously
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_e0_hw_r(
        nuscar._global_pool, plaintext, delta0, d0, list(guesses)
    )


def compute_not_e1_and_f0(plaintext: np.ndarray, delta0: np.ndarray, padded_key: bytes):
    """
    Compute the XOR of ~e1 (after first round compression) and initial f0.

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        ~e1 XOR f0 as 32-bit integer, shape (n,) or scalar
    """
    a0, b0, c, d0, e0, f0, g, h_val = compute_hi_ho(padded_key)
    t1 = compute_t10(plaintext, delta0)
    e1 = (d0 + t1) & 0xFFFFFFFF
    return ~e1 ^ f0


def compute_not_e1_and_f0_hw(plaintext: np.ndarray, delta0: np.ndarray, key: bytes):
    res = compute_not_e1_and_f0(plaintext, delta0, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_f0_hw(plaintext: np.ndarray, delta0: np.ndarray, d0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Attack all 4 bytes of f0 at once.
    Compute the Hamming weight of (~e1 XOR f0), where e1 = d0 + t10.

    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        d0: d0 value, numpy uint8 array, shape (4,)
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 4), attacking all 4 bytes simultaneously
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_f0_hw_r(
        nuscar._global_pool, plaintext, delta0, d0, list(guesses)
    )


def compute_t11(plaintext: np.ndarray, delta0: np.ndarray, padded_key: bytes):
    """
    Compute the t1 value in the second round compression of the second block of HMAC-SHA256.
    After the first round, the SHA-256 state changes:
    - e1 = d0 + t10
    - f1 = e0
    - g1 = f0
    - h1 = g0
    Second round t1 = h1 + Sigma1(e1) + Ch(e1, f1, g1) + K[1] + W[1]
                     = g0 + Sigma1(e1) + Ch(e1, e0, f0) + K[1] + w1
    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)
    Returns:
        t11 value, numpy uint8 array, shape (n, 4) or (4,)
    """
    w1 = compute_w1(plaintext)
    w1_32bit = uint8_array_to_uint32(w1)
    a0, b0, c0, d0, e0, f0, g0, h0 = compute_hi_ho(padded_key)
    t10 = compute_t10(plaintext, delta0)
    e1 = (d0 + t10) & 0xFFFFFFFF
    t11 = (g0 + big_sigma1(e1) + ch(e1, e0, f0) + _ROUND_K[1] + w1_32bit) & 0xFFFFFFFF
    return t11


def compute_t11_hw(plaintext: np.ndarray, delta0: np.ndarray, key: bytes):
    res = compute_t11(plaintext, delta0, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_g0_hw(plaintext: np.ndarray, delta0: np.ndarray, d0: np.ndarray, e0: np.ndarray, f0: np.ndarray,
                 byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Generic g0 byte attack function. Compute the Hamming weight of the target byte
    of t11 = g0 + Sigma1(e1) + Ch(e1, e0, f0) + K[1] + w1
    using known e1, w1 and guessed g0 bytes.
    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        d0: d0 value, numpy uint8 array, shape (4,)
        e0: e0 value, numpy uint8 array, shape (4,)
        f0: f0 value, numpy uint8 array, shape (4,)
        byte_index: byte index to attack (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB, e.g. [byte3, byte2, byte1]
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_g0_hw_r(
        nuscar._global_pool, plaintext, delta0, d0, e0, f0, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def attack_h0(delta0: np.ndarray, e0: np.ndarray, f0: np.ndarray, g0: np.ndarray) -> np.ndarray:
    """
    Recover h0 from known delta0, e0, f0, g0.
    Formula: delta0 = h0 + Sigma1(e0) + Ch(e0, f0, g0) + K[0]
    Inverse: h0 = delta0 - Sigma1(e0) - Ch(e0, f0, g0) - K[0]
    Subtraction performed modulo 2^32.

    Args:
        delta0: delta0 value, numpy uint8 array, shape (4,)
        e0: e0 value, numpy uint8 array, shape (4,)
        f0: f0 value, numpy uint8 array, shape (4,)
        g0: g0 value, numpy uint8 array, shape (4,)
    Returns:
        h0 value, numpy uint8 array, shape (4,)
    """
    delta0_32bit = int(uint8_array_to_uint32(delta0))
    e0_32bit = int(uint8_array_to_uint32(e0))
    f0_32bit = int(uint8_array_to_uint32(f0))
    g0_32bit = int(uint8_array_to_uint32(g0))

    h0 = (delta0_32bit - big_sigma1(e0_32bit) - ch(e0_32bit, f0_32bit, g0_32bit) - _ROUND_K[0]) & 0xFFFFFFFF

    return uint32_to_uint8_array(h0)


def compute_e2(plaintext: np.ndarray, delta0: np.ndarray, padded_key: bytes):
    """
    Compute the e value after the second round compression of the second block of HMAC-SHA256.

    In SHA-256, each round updates: e_new = d_old + t1
    - Round 1: e1 = d0 + t10
    - Round 2: e2 = d1 + t11 = c0 + t11 (since d1 = c0)

    Args:
        plaintext: 2D numpy uint8 array, shape (n, m)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        padded_key: pad-processed key, bytes type, generated by compute_padded_key(key, 0x36)

    Returns:
        e2 value as 32-bit integer, shape (n,) or scalar
    """
    w1 = compute_w1(plaintext)
    w1_32bit = uint8_array_to_uint32(w1)
    a0, b0, c0, d0, e0, f0, g0, h0 = compute_hi_ho(padded_key)

    # Round 1
    t10 = compute_t10(plaintext, delta0)
    e1 = (d0 + t10) & 0xFFFFFFFF

    # Round 2
    t11 = (g0 + big_sigma1(e1) + ch(e1, e0, f0) + _ROUND_K[1] + w1_32bit) & 0xFFFFFFFF
    e2 = (c0 + t11) & 0xFFFFFFFF

    return e2


def compute_e2_hw(plaintext: np.ndarray, delta0: np.ndarray, key: bytes):
    res = compute_e2(plaintext, delta0, key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_c0_hw(plaintext: np.ndarray, delta0: np.ndarray, d0: np.ndarray, e0: np.ndarray, f0: np.ndarray, g0: np.ndarray,
                 byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Generic c0 byte attack function.

    Compute the Hamming weight of the target byte of e2 = c0 + t11
    using known t11 and guessed c0 bytes.
    where t11 = g0 + Sigma1(e1) + Ch(e1, e0, f0) + K[1] + w1

    Args:
        plaintext: array of shape (N, ...)
        delta0: delta0 value, numpy uint8 array, shape (4,) or (n, 4)
        d0: d0 value, numpy uint8 array, shape (4,)
        e0: e0 value, numpy uint8 array, shape (4,)
        f0: f0 value, numpy uint8 array, shape (4,)
        g0: g0 value, numpy uint8 array, shape (4,)
        byte_index: byte index to attack (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB, e.g. [byte3, byte2, byte1]
        guesses: guess value array, default 0-255

    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sha256_attack_c0_hw_r(
        nuscar._global_pool, plaintext, delta0, d0, e0, f0, g0, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def sha256_expand_message(block: bytes) -> list:
    """
    Expand a 512-bit message block into a message schedule array of 64 32-bit words.

    Args:
        block: 64-byte message block (bytes).
    Returns:
        a list of 64 32-bit words (list).
    """
    return list(_n_rust.sha256_expand_message_r(list(block)))



def sha256(data, iv: list = None):
    """
    SHA-256 hash algorithm with optional custom initial vector (IV).
    Args:
        data: data to hash, can be:
            - bytes: byte data
            - np.ndarray (uint8): 1D or 2D array
              - 1D: shape (n,), returns a single hash
              - 2D: shape (m, n), returns m hashes
        iv: optional initial vector, list of 8 32-bit integers. If provided, used for HMAC computation.
    Returns:
        - If input is bytes or 1D array: returns a 32-byte bytes object
        - If input is 2D array: returns numpy uint8 array of shape (m, 32)
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return bytes(_n_rust.sha256_hash_r(list(data.tobytes()), iv))
        elif data.ndim == 2:
            data = np.ascontiguousarray(data, dtype=np.uint8)
            return _n_rust.sha256_hash_batch_r(nuscar._global_pool, data, iv)
        else:
            raise ValueError("data array dimension must be 1 or 2")

    # bytes input
    return bytes(_n_rust.sha256_hash_r(list(data), iv))


def hmac(key: bytes, msg: bytes) -> bytes:
    return bytes(_n_rust.sha256_hmac_r(list(key), list(msg)))


# --- Usage example ---
if __name__ == '__main__':
    hmac_key = '000102030405060708090a0b0c0d0e0f'
    message = 'ac7f92668c87d48b273bf8b62b145606'

    hmac_result = hmac(bytes.fromhex(hmac_key), bytes.fromhex(message))
    print(hmac_result.hex())



