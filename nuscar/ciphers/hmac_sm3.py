# Copyright(c)  2026. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as np
import nuscar
import nuscar.nuscar_rust as _n_rust

BLOCK_SIZE = 64

# SM3 initial value IV
_INITIAL_HASH = [
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e,
]

# SM3 constant T
_ROUND_T = [0x79cc4519] * 16 + [0x7a879d8a] * 48

def rotl(x, n):
    """Circular left shift"""
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def xor3(x, y, z):
    """XOR function: used by FF and GG in rounds 0-15"""
    return x ^ y ^ z


def maj(x, y, z):
    """Maj function: used by FF in rounds 16-63"""
    return (x & y) | (x & z) | (y & z)


def ch(x, y, z):
    """Ch function: used by GG in rounds 16-63"""
    return (x & y) | (~x & z)


def p0(x):
    """SM3 permutation function P0"""
    return x ^ rotl(x, 9) ^ rotl(x, 17)


def _p0_np(x):
    """Numpy vectorized P0: x ^ rotl(x,9) ^ rotl(x,17), supports uint32 arrays"""
    x = np.asarray(x, dtype=np.uint32)
    return x ^ ((x << 9) | (x >> 23)) ^ ((x << 17) | (x >> 15))


def p1(x):
    """SM3 permutation function P1"""
    return x ^ rotl(x, 15) ^ rotl(x, 23)


def uint32_to_uint8_array(w):
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


def uint8_array_to_uint32(arr):
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


def _i64(x):
    """Convert numpy uint32 scalar/array to int64 to avoid overflow warnings in arithmetic."""
    if isinstance(x, np.ndarray):
        return x.astype(np.int64)
    return int(x)



def sm3_expand_message(block: bytes):
    """
    SM3 message expansion: expand a 64-byte block into W[0..67] and W'[0..63].

    Args:
        block: 64-byte message block
    Returns:
        tuple (W, W_prime) where W is list of 68 uint32, W_prime is list of 64 uint32
    """
    return _n_rust.sm3_expand_message_r(list(block))


def sm3(data, iv: list = None):
    """
    SM3 hash algorithm with optional custom initial vector (IV).

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
            return bytes(_n_rust.sm3_hash_r(list(data.tobytes()), iv))
        elif data.ndim == 2:
            data = np.ascontiguousarray(data, dtype=np.uint8)
            return _n_rust.sm3_hash_batch_r(nuscar._global_pool, data, iv)
        else:
            raise ValueError("data array dimension must be 1 or 2")

    # bytes input
    return bytes(_n_rust.sm3_hash_r(list(data), iv))


def hmac(key: bytes, msg: bytes) -> bytes:
    """
    HMAC-SM3 algorithm.

    Args:
        key: secret key
        msg: message
    Returns:
        32-byte HMAC value
    """
    inner_key = compute_padded_key(key, 0x36)
    outer_key = compute_padded_key(key, 0x5c)

    inner_hash = sm3(inner_key + msg)
    hmac_value = sm3(outer_key + inner_hash)
    return hmac_value


def compute_w(plaintext, w_index):
    """
    Compute message word W[w_index] for HMAC-SM3 inner hash.

    Args:
        plaintext: plaintext data, can be bytes or numpy uint8 array
        w_index: message word index (0-67)
    Returns:
        - If plaintext is bytes: returns a single 32-bit integer
        - If plaintext is 1D array: returns uint8 array of shape (4,)
        - If plaintext is 2D array: returns uint8 array of shape (N, 4)
    """
    if isinstance(plaintext, bytes):
        arr = np.frombuffer(plaintext, dtype=np.uint8).reshape(1, -1)
        result = _n_rust.sm3_compute_w_r(nuscar._global_pool, arr, w_index)
        return uint8_array_to_uint32(np.asarray(result)[0])

    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        arr = plaintext[np.newaxis, :]
        result = _n_rust.sm3_compute_w_r(nuscar._global_pool, arr, w_index)
        return np.asarray(result)[0]
    elif plaintext.ndim == 2:
        return _n_rust.sm3_compute_w_r(nuscar._global_pool, plaintext, w_index)


def compute_w_prime(plaintext, w_index):
    """
    Compute message word W'[w_index] for HMAC-SM3 inner hash.
    W'[j] = W[j] ^ W[j+4]

    Args:
        plaintext: plaintext data
        w_index: W' index (0-63)
    Returns:
        value of W'[w_index]
    """
    if isinstance(plaintext, bytes):
        arr = np.frombuffer(plaintext, dtype=np.uint8).reshape(1, -1)
        result = _n_rust.sm3_compute_w_prime_r(nuscar._global_pool, arr, w_index)
        return uint8_array_to_uint32(np.asarray(result)[0])

    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        arr = plaintext[np.newaxis, :]
        result = _n_rust.sm3_compute_w_prime_r(nuscar._global_pool, arr, w_index)
        return np.asarray(result)[0]
    elif plaintext.ndim == 2:
        return _n_rust.sm3_compute_w_prime_r(nuscar._global_pool, plaintext, w_index)


def compute_padded_key(key: bytes, pad: int) -> bytes:
    """
    Compute HMAC padded key.

    Args:
        key: original key, bytes type
        pad: padding byte value, 0x36 (ipad) or 0x5c (opad)
    Returns:
        padded and XORed key, length is BLOCK_SIZE (64 bytes)
    """
    # If key length exceeds BLOCK_SIZE, hash it first
    if len(key) > BLOCK_SIZE:
        key = sm3(key)

    # Pad to BLOCK_SIZE
    if len(key) < BLOCK_SIZE:
        key = key + b'\x00' * (BLOCK_SIZE - len(key))

    # XOR with pad
    pad_bytes = bytes([pad] * BLOCK_SIZE)
    padded_key = bytes(k ^ p for k, p in zip(key, pad_bytes))
    return padded_key


def compute_hi_ho(padded_key: bytes) -> list:
    """Compute initial hash values for HMAC-SM3 inner/outer hash (after pad processing)"""
    return _n_rust.sm3_compute_hi_ho_r(list(padded_key))


def compute_a0_to_h0(padded_key: bytes) -> dict:
    """
    Compute all state values A0 to H0 after compressing the first block (ipad/opad) of HMAC-SM3.

    Args:
        padded_key: pad-processed key, bytes type
    Returns:
        dict containing A0 to H0, each value is a numpy uint8 array of shape (4,), big-endian
    """
    A0, B0, C0, D0, E0, F0, G0, H0 = compute_hi_ho(padded_key)
    return {
        'A0': uint32_to_uint8_array(A0),
        'B0': uint32_to_uint8_array(B0),
        'C0': uint32_to_uint8_array(C0),
        'D0': uint32_to_uint8_array(D0),
        'E0': uint32_to_uint8_array(E0),
        'F0': uint32_to_uint8_array(F0),
        'G0': uint32_to_uint8_array(G0),
        'H0': uint32_to_uint8_array(H0),
    }


def compute_ss1_0(padded_key: bytes) -> np.ndarray:
    """
    Compute SS1 value for SM3 round 0 (j=0).
    SS1_0 = ROTL((ROTL(A0,12) + E0 + ROTL(T0, 0)) & 0xFFFFFFFF, 7)

    Args:
        padded_key: pad-processed key
    Returns:
        4-byte representation of SS1_0, numpy uint8 array of shape (4,)
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    T0 = _ROUND_T[0]
    SS1_0 = rotl((rotl(A, 12) + E + rotl(T0, 0)) & 0xFFFFFFFF, 7)
    return uint32_to_uint8_array(SS1_0)


def compute_ss2_0(padded_key: bytes) -> np.ndarray:
    """
    Compute SS2 value for SM3 round 0 (j=0).
    SS2_0 = SS1_0 ^ ROTL(A0, 12)

    Args:
        padded_key: pad-processed key
    Returns:
        4-byte representation of SS2_0
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS1_0_32 = uint8_array_to_uint32(compute_ss1_0(padded_key))
    SS2_0 = SS1_0_32 ^ rotl(A, 12)
    return uint32_to_uint8_array(SS2_0)


def compute_delta1_0(padded_key: bytes) -> np.ndarray:
    """
    Compute HMAC-SM3 delta1_0 value. delta1_0 = xor3(A0,B0,C0) + D0 + SS2_0

    Args:
        padded_key: pad-processed key
    Returns:
        4-byte representation of delta1_0
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS2_0_32 = int(uint8_array_to_uint32(compute_ss2_0(padded_key)))
    xor_abc = xor3(A, B, C)
    delta1_0 = (xor_abc + D + SS2_0_32) & 0xFFFFFFFF
    return uint32_to_uint8_array(delta1_0)


def compute_delta2_0(padded_key: bytes) -> np.ndarray:
    """
    Compute HMAC-SM3 delta2_0 value. delta2_0 = xor3(E0,F0,G0) + H0 + SS1_0

    Args:
        padded_key: pad-processed key
    Returns:
        4-byte representation of delta2_0
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS1_0_32 = int(uint8_array_to_uint32(compute_ss1_0(padded_key)))
    xor_efg = xor3(E, F, G)
    delta2_0 = (xor_efg + H + SS1_0_32) & 0xFFFFFFFF
    return uint32_to_uint8_array(delta2_0)


def compute_tt1_0(plaintext: np.ndarray, delta1_0: np.ndarray):
    """
    Compute TT1 value for SM3 round 0. TT1_0 = delta1_0 + W'[0]
    """
    W_prime0 = compute_w_prime(plaintext, 0)
    W_prime0_32 = _i64(uint8_array_to_uint32(W_prime0))
    delta10_32 = _i64(uint8_array_to_uint32(delta1_0))
    TT1_0 = (delta10_32 + W_prime0_32) & 0xFFFFFFFF
    return TT1_0


def compute_tt10_hw(plaintext: np.ndarray, delta1_0: np.ndarray):
    res = compute_tt1_0(plaintext, delta1_0)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def compute_tt2_0(plaintext: np.ndarray, delta2_0: np.ndarray):
    """
    Compute TT2 value for SM3 round 0. TT2_0 = delta2_0 + W[0]
    """
    W0 = compute_w(plaintext, 0)
    W0_32 = _i64(uint8_array_to_uint32(W0))
    delta20_32 = _i64(uint8_array_to_uint32(delta2_0))
    TT2_0 = (delta20_32 + W0_32) & 0xFFFFFFFF
    return TT2_0


def compute_tt20_hw(plaintext: np.ndarray, delta2_0: np.ndarray):
    res = compute_tt2_0(plaintext, delta2_0)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_delta1_0_hw(plaintext: np.ndarray, byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 1: Generic byte-by-byte attack on delta1_0. TT1_0 = delta1_0 + W'[0]

    Args:
        plaintext: array of shape (N, ...)
        byte_index: target byte index (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes, ordered from LSB to MSB
        guesses: guess value array, default 0-255
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sm3_attack_delta1_0_hw_r(
        nuscar._global_pool, plaintext, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def attack_delta2_0_hw(plaintext: np.ndarray, byte_index: int, known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 2: Generic byte-by-byte attack on delta2_0. TT2_0 = delta2_0 + W[0]

    Args:
        plaintext: array of shape (N, ...)
        byte_index: target byte index (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    return _n_rust.sm3_attack_delta2_0_hw_r(
        nuscar._global_pool, plaintext, byte_index,
        list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def compute_tt11_a1_xor_b1(plaintext: np.ndarray, padded_key: bytes):
    """
    Compute leakage value of TT1_0 ^ A0 (for verifying step 3 attack).
    This function is used to test and verify the correctness of step 3 (attack on a0).

    Args:
        plaintext: plaintext data
        padded_key: padded key (only for verification, unknown in actual attack)
    Returns:
        32-bit integer value of TT1_0 ^ A0
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS2_0_32 = int(uint8_array_to_uint32(compute_ss2_0(padded_key)))
    xor_abc = xor3(A, B, C)
    delta1_0 = (xor_abc + D + SS2_0_32) & 0xFFFFFFFF
    tt10 = compute_tt1_0(plaintext, uint32_to_uint8_array(np.uint32(delta1_0)))
    return tt10 ^ A


def compute_tt11_a1_xor_b1_hw(plaintext: np.ndarray, padded_key: bytes):
    res = compute_tt11_a1_xor_b1(plaintext, padded_key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_a0_hw(plaintext: np.ndarray, delta1_0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 3: Generic attack on all 4 bytes of a0. Leakage: TT1_0 ^ a0

    Args:
        plaintext: array of shape (N, ...)
        delta1_0: recovered delta1_0 value
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 4)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta1_0 = np.asarray(delta1_0, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_a0_hw_r(
        nuscar._global_pool, plaintext, delta1_0, list(guesses)
    )


def compute_tt11_xor3(plaintext: np.ndarray, padded_key: bytes):
    """
    Compute leakage value of TT1_0 ^ A0 ^ rotl9(B0) (for verifying step 4 attack).
    This function is used to test and verify the correctness of step 4 (attack on rotl9(b0)).

    Args:
        plaintext: plaintext data
        padded_key: padded key (only for verification, unknown in actual attack)
    Returns:
        32-bit integer value of TT1_0 ^ A0 ^ rotl9(B0)
    """
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS2_0_32 = int(uint8_array_to_uint32(compute_ss2_0(padded_key)))
    xor_abc = xor3(A, B, C)
    delta1_0 = (xor_abc + D + SS2_0_32) & 0xFFFFFFFF
    tt10 = compute_tt1_0(plaintext, uint32_to_uint8_array(np.uint32(delta1_0)))
    return tt10 ^ A ^ rotl(B, 9)


def compute_tt11_xor3_hw(plaintext: np.ndarray, padded_key: bytes):
    res = compute_tt11_xor3(plaintext, padded_key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_b0_rotl9_hw(plaintext: np.ndarray, delta1_0: np.ndarray, a0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 4: Generic attack on all 4 bytes of rotl9(b0). Leakage: TT1_0 ^ a0 ^ rotl9(b0)

    Args:
        plaintext: array of shape (N, ...)
        delta1_0: recovered delta1_0 value
        a0: recovered a0 value
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 4)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta1_0 = np.asarray(delta1_0, dtype=np.uint8).ravel()
    a0 = np.asarray(a0, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_b0_rotl9_hw_r(
        nuscar._global_pool, plaintext, delta1_0, a0, list(guesses)
    )


def compute_tt21_e1_xor_f1(plaintext: np.ndarray, padded_key: bytes):
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS1_0_32 = int(uint8_array_to_uint32(compute_ss1_0(padded_key)))
    xor_efg = xor3(E, F, G)
    delta2_0 = (xor_efg + H + SS1_0_32) & 0xFFFFFFFF
    tt20 = compute_tt2_0(plaintext, uint32_to_uint8_array(np.uint32(delta2_0)))
    return p0(tt20) ^ E


def compute_tt21_e1_xor_f1_hw(plaintext: np.ndarray, padded_key: bytes):
    res = compute_tt21_e1_xor_f1(plaintext, padded_key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_e0_hw(plaintext: np.ndarray, delta2_0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 5: Generic attack on all 4 bytes of e0. Leakage: P0(TT2_0) ^ e0

    Args:
        plaintext: array of shape (N, ...)
        delta2_0: recovered delta2_0 value
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 4)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta2_0 = np.asarray(delta2_0, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_e0_hw_r(
        nuscar._global_pool, plaintext, delta2_0, list(guesses)
    )


def compute_tt21_xor3(plaintext: np.ndarray, padded_key: bytes):
    A, B, C, D, E, F, G, H = compute_hi_ho(padded_key)
    SS1_0_32 = int(uint8_array_to_uint32(compute_ss1_0(padded_key)))
    xor_efg = xor3(E, F, G)
    delta2_0 = (xor_efg + H + SS1_0_32) & 0xFFFFFFFF
    tt20 = compute_tt2_0(plaintext, uint32_to_uint8_array(np.uint32(delta2_0)))
    return p0(tt20) ^ E ^ rotl(F, 19)


def compute_tt21_xor3_hw(plaintext: np.ndarray, padded_key: bytes):
    res = compute_tt21_xor3(plaintext, padded_key)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_f0_rotl19_hw(plaintext: np.ndarray, delta2_0: np.ndarray, e0: np.ndarray, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 6: Generic attack on all 4 bytes of rotl19(f0). Leakage: P0(TT2_0) ^ e0 ^ rotl19(f0)

    Args:
        plaintext: array of shape (N, ...)
        delta2_0: recovered delta2_0 value
        e0: recovered e0 value
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 4)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta2_0 = np.asarray(delta2_0, dtype=np.uint8).ravel()
    e0 = np.asarray(e0, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_f0_rotl19_hw_r(
        nuscar._global_pool, plaintext, delta2_0, e0, list(guesses)
    )



def compute_tt11(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray, a0: np.ndarray, b0_rotl9: np.ndarray):
    w_prime1 = compute_w_prime(plaintext, 1)
    delta11 = compute_delta1_1(plaintext, delta1_0, delta2_0, a0, b0_rotl9)
    # delta11 is already uint32 (scalar or array), not uint8 array
    if isinstance(w_prime1, np.ndarray) and w_prime1.dtype == np.uint8:
        w_prime1_32 = uint8_array_to_uint32(w_prime1)
    else:
        w_prime1_32 = w_prime1
    return (delta11 + w_prime1_32) & 0xFFFFFFFF


def compute_tt11_hw(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray, a0: np.ndarray, b0_rotl9: np.ndarray):
    res = compute_tt11(plaintext, delta1_0, delta2_0, a0, b0_rotl9)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_c0_hw(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray,
                 a0: np.ndarray, b0_rotl9: np.ndarray, byte_index: int,
                 known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 7: Generic byte-by-byte attack on c0. TT1_1 = delta1_1 + C0 + W'[1]

    This is a byte-by-byte attack (4 iterations) due to addition carry propagation.

    Args:
        plaintext: array of shape (N, ...)
        delta1_0: recovered delta1_0
        delta2_0: recovered delta2_0
        a0: recovered a0
        b0_rotl9: recovered rotl9(b0)
        byte_index: target byte index (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes [byte3, byte2, ...] (LSB to MSB)
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta1_0 = np.asarray(delta1_0, dtype=np.uint8).ravel()
    delta2_0 = np.asarray(delta2_0, dtype=np.uint8).ravel()
    a0 = np.asarray(a0, dtype=np.uint8).ravel()
    b0_rotl9 = np.asarray(b0_rotl9, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_c0_hw_r(
        nuscar._global_pool, plaintext,
        delta1_0, delta2_0, a0, b0_rotl9,
        byte_index, list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


def compute_tt21(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray, a0: np.ndarray, f0_rotl9: np.ndarray):
    w_1 = compute_w(plaintext, 1)
    delta21 = compute_delta2_1(plaintext, delta1_0, delta2_0, a0, f0_rotl9)
    # delta21 is already uint32 (scalar or array), not uint8 array
    if isinstance(w_1, np.ndarray) and w_1.dtype == np.uint8:
        w1_32 = uint8_array_to_uint32(w_1)
    else:
        w1_32 = w_1
    return (delta21 + w1_32) & 0xFFFFFFFF


def compute_tt21_hw(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray, a0: np.ndarray, f0_rotl9: np.ndarray):
    res = compute_tt21(plaintext, delta1_0, delta2_0, a0, f0_rotl9)
    return nuscar.leakmodel.leakage_model_hw(uint32_to_uint8_array(res), nb_words=1)


def attack_g0_hw(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray,
                 e0: np.ndarray, f0_rotl19: np.ndarray, byte_index: int,
                 known_bytes=None, guesses=np.arange(256, dtype=np.uint8)):
    """
    Step 8: Generic byte-by-byte attack on g0. TT2_1 = delta2_1 + G0 + W[1]
    This is a byte-by-byte attack (4 iterations) due to addition carry propagation.

    Args:
        plaintext: array of shape (N, ...)
        delta1_0: recovered delta1_0
        delta2_0: recovered delta2_0
        e0: recovered e0
        f0_rotl19: recovered rotl19(f0)
        byte_index: target byte index (0=MSB, 3=LSB)
        known_bytes: list of known lower bytes [byte3, byte2, ...] (LSB to MSB)
        guesses: guess value array
    Returns:
        Hamming weight array of shape (N, 256, 1)
    """
    plaintext = np.asarray(plaintext, dtype=np.uint8)
    if plaintext.ndim == 1:
        plaintext = plaintext[np.newaxis, :]
    delta1_0 = np.asarray(delta1_0, dtype=np.uint8).ravel()
    delta2_0 = np.asarray(delta2_0, dtype=np.uint8).ravel()
    e0 = np.asarray(e0, dtype=np.uint8).ravel()
    f0_rotl19 = np.asarray(f0_rotl19, dtype=np.uint8).ravel()
    return _n_rust.sm3_attack_g0_hw_r(
        nuscar._global_pool, plaintext,
        delta1_0, delta2_0, e0, f0_rotl19,
        byte_index, list(guesses),
        list(known_bytes) if known_bytes is not None else None
    )


# ============ Round 2 intermediate value computation functions ============
def compute_ss1_1(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray) -> int:
    """
    Compute SS1 value for SM3 round 1 (j=1).

    State after round 0:
    A = TT1_0 = delta1_0 + W'[0]
    E = P0(TT2_0) = P0(delta2_0 + W[0])

    SS1_1 = rotl((rotl(A, 12) + E + rotl(T[1], 1)) & 0xFFFFFFFF, 7)

    Args:
        plaintext: plaintext data
        delta1_0: recovered delta1_0
        delta2_0: recovered delta2_0
    Returns:
        uint32 representation of SS1_1
    """
    # Compute A = TT1_0
    W_prime0 = compute_w_prime(plaintext, 0)
    W_prime0_32 = _i64(uint8_array_to_uint32(W_prime0))
    delta1_0_32 = _i64(uint8_array_to_uint32(delta1_0))
    A = (delta1_0_32 + W_prime0_32) & 0xFFFFFFFF

    # Compute E = P0(TT2_0)
    W0 = compute_w(plaintext, 0)
    W0_32 = _i64(uint8_array_to_uint32(W0))
    delta2_0_32 = _i64(uint8_array_to_uint32(delta2_0))
    TT2_0 = (delta2_0_32 + W0_32) & 0xFFFFFFFF

    E = _p0_np(TT2_0)

    # Compute SS1_1
    T1 = _ROUND_T[1]
    SS1_1 = rotl((_i64(rotl(A, 12)) + _i64(E) + rotl(T1, 1)) & 0xFFFFFFFF, 7)
    return SS1_1


def compute_ss2_1(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray) -> int:
    """
    Compute SS2 value for SM3 round 1 (j=1).
    SS2_1 = SS1_1 ^ rotl(A, 12)

    where A = TT1_0

    Args:
        plaintext: plaintext data
        delta1_0: recovered delta1_0
        delta2_0: recovered delta2_0
    Returns:
        uint32 representation of SS2_1
    """
    # Compute A = TT1_0
    W_prime0 = compute_w_prime(plaintext, 0)
    W_prime0_32 = _i64(uint8_array_to_uint32(W_prime0))
    delta1_0_32 = _i64(uint8_array_to_uint32(delta1_0))
    A = (delta1_0_32 + W_prime0_32) & 0xFFFFFFFF

    # Compute SS1_1
    SS1_1_32 = compute_ss1_1(plaintext, delta1_0, delta2_0)

    # Compute SS2_1
    SS2_1 = SS1_1_32 ^ rotl(A, 12)
    return SS2_1


def compute_delta1_1(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray,
                     a0: np.ndarray, b0_rotl9: np.ndarray) -> int:
    """
    Compute delta1_1 for round 1.

    State after round 0:
    A = TT1_0
    B = A0
    C = rotl(B0, 9)

    delta1_1 = xor3(A, B, C) + SS2_1
             = xor3(TT1_0, A0, rotl9(B0)) + SS2_1

    Then TT1_1 = delta1_1 + D + W'[1] = delta1_1 + C0 + W'[1]

    Args:
        plaintext: plaintext data
        delta1_0, delta2_0: recovered delta values
        a0: recovered a0
        b0_rotl9: recovered rotl9(b0)
    Returns:
        4-byte representation of delta1_1
    """
    # Compute A = TT1_0
    W_prime0 = compute_w_prime(plaintext, 0)
    W_prime0_32 = _i64(uint8_array_to_uint32(W_prime0))
    delta1_0_32 = _i64(uint8_array_to_uint32(delta1_0))
    A = (delta1_0_32 + W_prime0_32) & 0xFFFFFFFF

    # B = A0, C = rotl9(B0)
    a0_32 = _i64(uint8_array_to_uint32(a0))
    b0_rotl9_32 = _i64(uint8_array_to_uint32(b0_rotl9))

    # Compute SS2_1
    SS2_1_32 = compute_ss2_1(plaintext, delta1_0, delta2_0)

    # delta1_1 = xor3(A, B, C) + SS2_1
    xor_abc = xor3(A, a0_32, b0_rotl9_32)
    delta1_1 = (xor_abc + _i64(SS2_1_32)) & 0xFFFFFFFF
    return delta1_1


def compute_delta2_1(plaintext: np.ndarray, delta1_0: np.ndarray, delta2_0: np.ndarray,
                     e0: np.ndarray, f0_rotl19: np.ndarray) -> int:
    """
    Compute delta2_1 for round 1.

    State after round 0:
    E = P0(TT2_0)
    F = E0
    G = rotl(F0, 19)

    delta2_1 = xor3(E, F, G) + SS1_1
             = xor3(P0(TT2_0), E0, rotl19(F0)) + SS1_1

    Then TT2_1 = delta2_1 + H + W[1] = delta2_1 + G0 + W[1]

    Args:
        plaintext: plaintext data
        delta1_0, delta2_0: recovered delta values
        e0: recovered e0
        f0_rotl19: recovered rotl19(f0)
    Returns:
        4-byte representation of delta2_1
    """
    # Compute E = P0(TT2_0)
    W0 = compute_w(plaintext, 0)
    W0_32 = _i64(uint8_array_to_uint32(W0))
    delta2_0_32 = _i64(uint8_array_to_uint32(delta2_0))
    TT2_0 = (delta2_0_32 + W0_32) & 0xFFFFFFFF

    E = _p0_np(TT2_0)

    # F = E0, G = rotl19(F0)
    e0_32 = _i64(uint8_array_to_uint32(e0))
    f0_rotl19_32 = _i64(uint8_array_to_uint32(f0_rotl19))

    # Compute SS1_1
    SS1_1_32 = compute_ss1_1(plaintext, delta1_0, delta2_0)

    # delta2_1 = xor3(E, F, G) + SS1_1
    xor_efg = _i64(xor3(E, e0_32, f0_rotl19_32))
    delta2_1 = (xor_efg + _i64(SS1_1_32)) & 0xFFFFFFFF
    return delta2_1


# ============ Recovery functions: compute remaining state from attacked values ============
def recover_b0_from_rotl9(b0_rotl9: np.ndarray) -> np.ndarray:
    b0_rotl9_32 = uint8_array_to_uint32(b0_rotl9)
    b0 = rotl(b0_rotl9_32, 23)
    return uint32_to_uint8_array(b0)


def recover_f0_from_rotl19(f0_rotl19: np.ndarray) -> np.ndarray:
    """Recover f0 from rotl19(f0)"""
    f0_rotl19_32 = uint8_array_to_uint32(f0_rotl19)
    f0 = rotl(f0_rotl19_32, 13)
    return uint32_to_uint8_array(f0)


def recover_d0(delta1_0: np.ndarray, a0: np.ndarray, b0: np.ndarray, c0: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """
    Step 9: Recover d0. d0 = delta1_0 - xor3(a0, b0, c0) - SS2_0

    where SS2_0 can be computed directly from recovered a0 and e0:
    SS1_0 = rotl((rotl(a0, 12) + e0 + rotl(T[0], 0)) & 0xFFFFFFFF, 7)
    SS2_0 = SS1_0 ^ rotl(a0, 12)

    Args:
        delta1_0: recovered delta1_0
        a0, b0, c0: recovered state values
        e0: recovered e0 (used to compute SS2_0)
    Returns:
        4-byte representation of d0
    """
    delta1_0_32 = int(uint8_array_to_uint32(delta1_0))
    a0_32 = int(uint8_array_to_uint32(a0))
    b0_32 = int(uint8_array_to_uint32(b0))
    c0_32 = int(uint8_array_to_uint32(c0))
    e0_32 = int(uint8_array_to_uint32(e0))

    # Compute SS1_0 and SS2_0
    T0 = _ROUND_T[0]
    SS1_0 = rotl((rotl(a0_32, 12) + e0_32 + rotl(T0, 0)) & 0xFFFFFFFF, 7)
    SS2_0 = SS1_0 ^ rotl(a0_32, 12)

    xor_abc = xor3(a0_32, b0_32, c0_32)
    d0 = (delta1_0_32 - xor_abc - SS2_0) & 0xFFFFFFFF

    return uint32_to_uint8_array(d0)


def recover_h0(delta2_0: np.ndarray, a0: np.ndarray, e0: np.ndarray, f0: np.ndarray, g0: np.ndarray) -> np.ndarray:
    """
    Step 10: Recover h0. h0 = delta2_0 - xor3(e0, f0, g0) - SS1_0

    where SS1_0 can be computed directly from recovered a0 and e0:
    SS1_0 = rotl((rotl(a0, 12) + e0 + rotl(T[0], 0)) & 0xFFFFFFFF, 7)

    Args:
        delta2_0: recovered delta2_0
        a0: recovered a0 (used to compute SS1_0)
        e0, f0, g0: recovered state values
    Returns:
        4-byte representation of h0
    """
    delta2_0_32 = int(uint8_array_to_uint32(delta2_0))
    a0_32 = int(uint8_array_to_uint32(a0))
    e0_32 = int(uint8_array_to_uint32(e0))
    f0_32 = int(uint8_array_to_uint32(f0))
    g0_32 = int(uint8_array_to_uint32(g0))

    # Compute SS1_0
    T0 = _ROUND_T[0]
    SS1_0 = rotl((rotl(a0_32, 12) + e0_32 + rotl(T0, 0)) & 0xFFFFFFFF, 7)

    xor_efg = xor3(e0_32, f0_32, g0_32)
    h0 = (delta2_0_32 - xor_efg - SS1_0) & 0xFFFFFFFF

    return uint32_to_uint8_array(h0)


def recover_sm3_state(delta1_0, delta2_0, a0, b0_rotl9, c0, e0, f0_rotl19, g0) -> dict:
    """
    Reconstruct the complete SM3 state from side-channel attack recovered values.

    This function does not depend on padded_key; it only uses values recovered from
    side-channel attacks.

    Args:
        delta1_0, delta2_0: recovered delta values
        a0, b0_rotl9, c0: recovered A/B/C state (B is rotated)
        e0, f0_rotl19, g0: recovered E/F/G state (F is rotated)
    Returns:
        dict containing the complete state {A0, B0, C0, D0, E0, F0, G0, H0}
    """
    # Recover b0 and f0
    b0 = recover_b0_from_rotl9(b0_rotl9)
    f0 = recover_f0_from_rotl19(f0_rotl19)

    # Compute d0 and h0 (does not require padded_key)
    d0 = recover_d0(delta1_0, a0, b0, c0, e0)
    h0 = recover_h0(delta2_0, a0, e0, f0, g0)

    return {
        'A0': a0,
        'B0': b0,
        'C0': c0,
        'D0': d0,
        'E0': e0,
        'F0': f0,
        'G0': g0,
        'H0': h0,
    }


# --- Usage examples ---
if __name__ == '__main__':
    # Test SM3
    print("=== SM3 Test ===")

    # Test 1: empty message
    msg1 = b""
    hash1 = sm3(msg1)
    print(f"SM3(\"\") = {hash1.hex()}")
    # Expected: 1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b

    # Test 2: "abc"
    msg2 = b"abc"
    hash2 = sm3(msg2)
    print(f"SM3(\"abc\") = {hash2.hex()}")
    # Expected: 66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0

    # Test 3: long message
    msg3 = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd"
    hash3 = sm3(msg3)
    print(f"SM3(long message) = {hash3.hex()}")
    # Expected: debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732

    print("\n=== HMAC-SM3 Test ===")

    # Test HMAC-SM3
    key = b"key"
    msg = b"The quick brown fox jumps over the lazy dog"
    hmac_result = hmac(key, msg)
    print(f"HMAC-SM3(key, msg) = {hmac_result.hex()}")

    # Test custom key
    key2 = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    msg2 = bytes.fromhex("ac7f92668c87d48b273bf8b62b145606")
    hmac_result2 = hmac(key2, msg2)
    print(f"HMAC-SM3(hex_key, hex_msg) = {hmac_result2.hex()}")
