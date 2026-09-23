# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from ._hwcaps import require_aes_support as _require_aes_support

_require_aes_support()

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from typing import Callable
import nuscar
from makefun import with_signature
from inspect import Signature, Parameter
import enum

SBOX = _np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16],
    dtype=_np.uint8
)

INV_SBOX = _np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D],
    dtype=_np.uint8)

RCON = _np.array([
    [0x01, 0x00, 0x00, 0x00],
    [0x02, 0x00, 0x00, 0x00],
    [0x04, 0x00, 0x00, 0x00],
    [0x08, 0x00, 0x00, 0x00],
    [0x10, 0x00, 0x00, 0x00],
    [0x20, 0x00, 0x00, 0x00],
    [0x40, 0x00, 0x00, 0x00],
    [0x80, 0x00, 0x00, 0x00],
    [0x1b, 0x00, 0x00, 0x00],
    [0x36, 0x00, 0x00, 0x00]],
    dtype=_np.uint8
)

SHIFT_ROWS = _np.array([0, 5, 10, 15, 4, 9, 14, 3, 8,
                       13, 2, 7, 12, 1, 6, 11], dtype=_np.uint8)
INV_SHIFT_ROWS = _np.array(
    [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3], dtype=_np.uint8)


class Steps(enum.IntEnum):
    """Enumeration for the four AES round steps."""

    SUB_BYTES = 0
    SHIFT_ROWS = 1
    MIX_COLUMNS = 2
    ADD_ROUND_KEY = 3


class InvSteps(enum.IntEnum):
    """Enumeration for the four inverse AES round steps."""

    INV_ADD_ROUND_KEY = 0
    INV_MIX_COLUMNS = 1
    INV_SHIFT_ROWS = 2
    INV_SUB_BYTES = 3


def sub_bytes(state):
    return SBOX[state]


def inv_sub_bytes(state):
    return INV_SBOX[state]


def shift_rows(state):
    """Apply AES ShiftRows transformation.

    Args:
        state: 1D array of 16 bytes or 2D array of shape (n, 16).

    Returns:
        Transformed array. 1D input returns 1D, 2D input returns 2D.
    """
    state = _np.asarray(state)
    squeeze = state.ndim == 1
    if state.ndim == 1:
        if state.shape[0] != 16:
            raise ValueError(f"1D input must have 16 bytes, got {state.shape[0]}")
        state = _np.expand_dims(state, axis=0)
    elif state.ndim == 2:
        if state.shape[1] != 16:
            raise ValueError(f"2D input must have shape (n, 16), got {state.shape}")
    else:
        raise ValueError(f"Input must be 1D or 2D array, got {state.ndim}D")

    result = state.astype(_np.uint8, copy=False)[:, SHIFT_ROWS]
    return result.squeeze(axis=0) if squeeze else result


def inv_shift_rows(state):
    """Apply AES InvShiftRows transformation.

    Args:
        state: 1D array of 16 bytes or 2D array of shape (n, 16).

    Returns:
        Transformed array. 1D input returns 1D, 2D input returns 2D.
    """
    state = _np.asarray(state)
    squeeze = state.ndim == 1
    if state.ndim == 1:
        if state.shape[0] != 16:
            raise ValueError(f"1D input must have 16 bytes, got {state.shape[0]}")
        state = _np.expand_dims(state, axis=0)
    elif state.ndim == 2:
        if state.shape[1] != 16:
            raise ValueError(f"2D input must have shape (n, 16), got {state.shape}")
    else:
        raise ValueError(f"Input must be 1D or 2D array, got {state.ndim}D")

    result = state.astype(_np.uint8, copy=False)[:, INV_SHIFT_ROWS]
    return result.squeeze(axis=0) if squeeze else result


def mix_columns(state):
    """Apply AES MixColumns transformation.

    Args:
        state: 1D array of 16 bytes or 2D array of shape (n, 16).

    Returns:
        Transformed array. 1D input returns 1D, 2D input returns 2D.
    """
    state = _np.asarray(state)
    squeeze = state.ndim == 1
    if state.ndim == 1:
        if state.shape[0] != 16:
            raise ValueError(f"1D input must have 16 bytes, got {state.shape[0]}")
        state = _np.expand_dims(state, axis=0)
    elif state.ndim == 2:
        if state.shape[1] != 16:
            raise ValueError(f"2D input must have shape (n, 16), got {state.shape}")
    else:
        raise ValueError(f"Input must be 1D or 2D array, got {state.ndim}D")

    result = _n_rust.aes_mixcolumns_r(nuscar._global_pool, state.astype(_np.uint8, copy=False))
    return result.squeeze(axis=0) if squeeze else result


def inv_mix_columns(state):
    """Apply AES InvMixColumns transformation.

    Args:
        state: 1D array of 16 bytes or 2D array of shape (n, 16).

    Returns:
        Transformed array. 1D input returns 1D, 2D input returns 2D.
    """
    state = _np.asarray(state)
    squeeze = state.ndim == 1
    if state.ndim == 1:
        if state.shape[0] != 16:
            raise ValueError(f"1D input must have 16 bytes, got {state.shape[0]}")
        state = _np.expand_dims(state, axis=0)
    elif state.ndim == 2:
        if state.shape[1] != 16:
            raise ValueError(f"2D input must have shape (n, 16), got {state.shape}")
    else:
        raise ValueError(f"Input must be 1D or 2D array, got {state.ndim}D")

    result = _n_rust.aes_inv_mixcolumns_r(nuscar._global_pool, state.astype(_np.uint8, copy=False))
    return result.squeeze(axis=0) if squeeze else result

def attack_first_addRk_hw(meta_name='plaintext', pos=list(range(16)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of the first AddRoundKey.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.xor_attack_hw_with_guess_r(nuscar._global_pool, plaintext[:, pos], kwargs["guesses"])

    return func_impl

def attack_first_addRk_value(meta_name='plaintext', pos=list(range(16)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of the first AddRoundKey.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.xor_attack_value_with_guess_r(nuscar._global_pool, plaintext[:, pos], kwargs["guesses"])

    return func_impl

def attack_first_addRk_bit(meta_name='plaintext', pos=list(range(16)), bit_pos=0, guesses=list(range(256))) -> Callable:
    """generate a selection function targeting one bit of the first AddRoundKey.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        bit_pos (int, optional): target bit, 0-7.
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    assert (7 >= bit_pos >= 0)
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
        "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
    ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.xor_attack_bit_with_guess_r(nuscar._global_pool, plaintext[:, pos], bit_pos, guesses)
    return func_impl

def compute_first_addRk_hw(plaintext, key):
    """Compute the hamming weight of the first AddRoundKey operation"""
    res = encrypt(plaintext, key, stop_round=0, stop_step=Steps.ADD_ROUND_KEY)
    return nuscar.leakmodel.leakage_model_hw(res)

def compute_first_addRk_value(plaintext, key):
    """Compute the value of the first AddRoundKey operation"""
    res = encrypt(plaintext, key, stop_round=0, stop_step=Steps.ADD_ROUND_KEY)
    return res

def attack_first_sbox_hw(meta_name='plaintext', pos=list(range(16)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of sbox out at first round.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_first_sbox_hw_with_guess_r(nuscar._global_pool, plaintext[:, pos], kwargs["guesses"])

    return func_impl

def attack_first_sbox_value(meta_name='plaintext', pos=list(range(16)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of sbox out at first round.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_first_sbox_value_with_guess_r(nuscar._global_pool, plaintext[:, pos], kwargs["guesses"])

    return func_impl

def attack_first_sbox_bit(meta_name='plaintext', pos=list(range(16)), bit_pos=0, guesses=list(range(256))) -> Callable:
    """generate a selection function targeting one bit of sbox out at first round.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        bit_pos (int, optional): target bit, 0-7.
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    assert (7 >= bit_pos >= 0)
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
        "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
    ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        plaintext = kwargs[meta_name]
        if len(plaintext.shape) == 1:
            plaintext = _np.expand_dims(plaintext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_first_sbox_bit_with_guess_r(nuscar._global_pool, plaintext[:, pos], bit_pos, guesses)
    return func_impl

def compute_first_sbox_hw(plaintext, key):
    """Compute the hamming weight of the first sbox operation"""
    res = encrypt(plaintext, key, stop_round=1, stop_step=Steps.SUB_BYTES)
    return nuscar.leakmodel.leakage_model_hw(res)

def compute_first_sbox_value(plaintext, key):
    """Compute the value of the first sbox operation"""
    res = encrypt(plaintext, key, stop_round=1, stop_step=Steps.SUB_BYTES)
    return res

def attack_last_sbox_hw(meta_name='ciphertext', pos=list(range(16)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of sbox input at last round.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'ciphertext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        ciphertext = kwargs[meta_name]
        if ciphertext.ndim == 1:
            ciphertext = _np.expand_dims(ciphertext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_last_sbox_hw_with_guess_r(nuscar._global_pool, ciphertext[:, pos], kwargs["guesses"])

    return func_impl


def attack_last_sbox_bit(meta_name='ciphertext', pos=list(range(16)), bit_pos=0, guesses=list(range(256))) -> Callable:
    """generate a selection function targeting one bit of sbox input at last round.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'ciphertext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        bit_pos (int, optional): target bit, 0-7.
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    assert (7 >= bit_pos >= 0)
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
        "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
    ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        ciphertext = kwargs[meta_name]
        if ciphertext.ndim == 1:
            ciphertext = _np.expand_dims(ciphertext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_last_sbox_bit_with_guess_r(nuscar._global_pool, ciphertext[:, pos], bit_pos, guesses)
    return func_impl

def compute_last_sbox_hw(ciphertext, key):
    """Compute the hamming weight of the last sbox operation"""
    res = decrypt(ciphertext, key, stop_round=1, stop_step=InvSteps.INV_SHIFT_ROWS)
    return nuscar.leakmodel.leakage_model_hw(res)

def compute_last_sbox_value(ciphertext, key):
    """Compute the value of the last sbox operation"""
    res = decrypt(ciphertext, key, stop_round=1, stop_step=InvSteps.INV_SHIFT_ROWS)
    return res


def attack_last_round_xor_hw(meta_name='ciphertext', pos=list(range(16)),  guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming distance of last round input and output.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'ciphertext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        bit_pos (int, optional): target bit, 0-7.
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
                      "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
                  ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        ciphertext = kwargs[meta_name]
        if ciphertext.ndim == 1:
            ciphertext = _np.expand_dims(ciphertext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_last_round_xor_hw_with_guess_r(nuscar._global_pool, ciphertext, pos, guesses)
    return func_impl


def attack_last_round_xor_bit(meta_name='ciphertext', pos=list(range(16)), bit_pos=0, guesses=list(range(256))) -> Callable:
    """generate a selection function targeting one bit xor of last round input and output.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'ciphertext'.
        pos (list, optional): key positions to attack. Defaults to list(range(16)).
        bit_pos (int, optional): target bit, 0-7.
        guesses (list, optional): guess range for each key. Defaults to list(range(256)).

    Returns:
        Callable: selection funtion
    """
    assert (7 >= bit_pos >= 0)
    parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                  Parameter(
        "guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses),
    ]
    func_sig = Signature(parameters)

    @with_signature(func_sig, func_name="func")
    def func_impl(**kwargs):
        ciphertext = kwargs[meta_name]
        if ciphertext.ndim == 1:
            ciphertext = _np.expand_dims(ciphertext, axis=0)  # keep 2D array
        return _n_rust.aes_attack_last_round_xor_bit_with_guess_r(nuscar._global_pool, ciphertext, bit_pos, pos, guesses)
    return func_impl


def compute_last_round_xor_hw(ciphertext, key):
    """Compute the hamming weight of the xor of last round input and output"""
    res = ciphertext ^ decrypt(ciphertext, key, stop_round=1, stop_step=InvSteps.INV_SUB_BYTES)
    return nuscar.leakmodel.leakage_model_hw(res)

def compute_last_round_xor_value(ciphertext, key):
    """Compute the value of the xor of last round input and output """
    res = ciphertext ^ decrypt(ciphertext, key, stop_round=1, stop_step=InvSteps.INV_SUB_BYTES)
    return res



_n_round = {16: 10, 24: 12, 32: 14}


def encrypt(plaintext: _np.ndarray, key: _np.ndarray, stop_round=None, stop_step: Steps = Steps.ADD_ROUND_KEY) -> _np.ndarray:
    """aes encrypt plaintext with given key, may return intermediate state at given stop_round and stop_step.

    Args:
        plaintext (ndarray): 1-D/2-D plaintext.
        key (ndarray): 1-D/2-D key.
        stop_round (int, optional): which round to stop. Defaults to 10. 
        stop_step (Steps, optional): which step to stop. Defaults to Steps.ADD_ROUND_KEY.
    """
    if stop_round is None:
        stop_round = _n_round[key.shape[-1]]
    if stop_round == 0 and stop_step != Steps.ADD_ROUND_KEY:
        raise ValueError("AES round %d has no %s step" %
                         (stop_round, str(stop_step)))
    if key.shape[-1] == 16:
        if stop_round == 10 and stop_step == Steps.MIX_COLUMNS:
            raise ValueError("AES-128 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 10:
            raise ValueError("AES-128 round number cannot exceed 10")
    if key.shape[-1] == 24:
        if stop_round == 12 and stop_step == Steps.MIX_COLUMNS:
            raise ValueError("AES-192 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 12:
            raise ValueError("AES-192 round number cannot exceed 12")
    if key.shape[-1] == 32:
        if stop_round == 14 and stop_step == Steps.MIX_COLUMNS:
            raise ValueError("AES-256 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 14:
            raise ValueError("AES-256 round number cannot exceed 14")

    nb_words = plaintext.shape[-1]
    plaintext = plaintext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.aes_encrypt_step_r(
            nuscar._global_pool, plaintext, key, stop_round, stop_step.value)
    elif (len(key.shape) == 1):
        return _n_rust.aes_encrypt_step_fix_key_r(
            nuscar._global_pool, plaintext, key, stop_round, stop_step.value)
    else:
        raise ValueError("key shape cannot be %s" % str(key.shape))


def decrypt(ciphertext: _np.ndarray, key: _np.ndarray, stop_round=None, stop_step: InvSteps = InvSteps.INV_ADD_ROUND_KEY) -> _np.ndarray:
    """aes decrypt ciphertext with given key, may return intermediate state at given stop_round and stop_step.

    Args:
        ciphertext (ndarray): 1-D/2-D plaintext.
        key (ndarray): 1-D/2-D key.
        stop_round (int, optional): which round to stop. Defaults to 10. 
        stop_step (Steps, optional): which step to stop. Defaults to InvSteps.INV_ADD_ROUND_KEY.
    """
    if stop_round is None:
        stop_round = _n_round[key.shape[-1]]
    if stop_round == 0 and stop_step != InvSteps.INV_ADD_ROUND_KEY:
        raise ValueError("AES round %d has no %s step" %
                         (stop_round, str(stop_step)))
    if key.shape[-1] == 16:
        if stop_round == 10 and stop_step == InvSteps.INV_MIX_COLUMNS:
            raise ValueError("AES-128 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 10:
            raise ValueError("AES-128 round number cannot exceed 10")
    if key.shape[-1] == 24:
        if stop_round == 12 and stop_step == InvSteps.INV_MIX_COLUMNS:
            raise ValueError("AES-192 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 12:
            raise ValueError("AES-192 round number cannot exceed 12")
    if key.shape[-1] == 32:
        if stop_round == 14 and stop_step == InvSteps.INV_MIX_COLUMNS:
            raise ValueError("AES-256 round %d has no %s step" %
                             (stop_round, str(stop_step)))
        if stop_round > 14:
            raise ValueError("AES-256 round number cannot exceed 14")
    nb_words = ciphertext.shape[-1]
    ciphertext = ciphertext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.aes_decrypt_step_r(
            nuscar._global_pool, ciphertext, key, stop_round, stop_step.value)
    elif (len(key.shape) == 1):
        return _n_rust.aes_decrypt_step_fix_key_r(
            nuscar._global_pool, ciphertext, key, stop_round, stop_step.value)
    else:
        raise ValueError("key shape cannot be %s" % str(key.shape))

def _is_bytes_of_len(state, length=[16]):
    if state.shape[-1] not in length:
        raise ValueError(f'state last dimension should be in {length}, not {state.shape[-1]}.')
    return True


def key_schedule(key):
    """Compute AES key schedules for any AES mode.

    Handle AES-128, AES-192, AES-256 modes, respectively 16, 24 and 32 bytes vector key length, given as numpy array input.

    Args:
        key (numpy.ndarray): numpy byte array (dtype uint8). Last dimension must be 16, 24, or 32 long, for resp. AES 128, 192 and 256 mode.

    Returns:
        (numpy.ndarray): numpy byte array containing all round keys, with shape (number of keys, number of rounds, 16), or (number of rounds, 16) if
            only one key has been provided.

    Examples:
        import numpy as np
        key = np.array([0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C], dtype=np.uint8)
        schedule = key_schedule(key)

    """
    keys = _key_expansion(key, col_in=0)
    if key.shape[:-1]:
        final_shape = (keys.shape[0], int(keys.shape[1] / 16), 16)
    else:
        final_shape = (int(keys.shape[1] / 16), 16)
    return keys.reshape(final_shape)


def inv_key_schedule(key, round_in=10):
    """Recover the full key schedule from round key bytes by reversing the key expansion.

    Given consecutive round key bytes starting at round round_in, recover the master key
    and compute the full key schedule. The input key length determines the AES mode:
      - 16 bytes: AES-128, needs 1 round key (round_in=10 for last round).
      - 24 bytes: AES-192, needs 1.5 round keys (round_in=12 for last round).
      - 32 bytes: AES-256, needs 2 round keys (round_in=14 for last 2 rounds).

    Args:
        key (numpy.ndarray): round key bytes (dtype uint8). Last dimension must be 16, 24, or 32.
        round_in (int): the round index where the input key bytes start (col_in = round_in * 4).
            Defaults to 10 (last round of AES-128).

    Returns:
        numpy.ndarray: the full key schedule, with shape (number of rounds + 1, 16).

    Examples:
        >>> import numpy as np
        >>> # AES-128: recover from last round key (round 10, 16 bytes)
        >>> rk = np.array([...], dtype=np.uint8)  # 16 bytes
        >>> full_ks = inv_key_schedule(rk, round_in=10)
        >>> # AES-192: recover from round 12 (24 bytes spanning last 1.5 rounds)
        >>> rk = np.array([...], dtype=np.uint8)  # 24 bytes
        >>> full_ks = inv_key_schedule(rk, round_in=12)
        >>> # AES-256: recover from round 14 (32 bytes spanning last 2 rounds)
        >>> rk = np.array([...], dtype=np.uint8)  # 32 bytes
        >>> full_ks = inv_key_schedule(rk, round_in=14)
    """
    key_len = key.shape[-1]
    expansion = _key_expansion(key, col_in=round_in * 4, col_out=0)
    masters = expansion.swapaxes(0, -1)[:key_len].swapaxes(0, -1)
    return _np.squeeze(key_schedule(masters))


# cols_out: number of columns of 4 bytes in the output expanded key
_cols_out = {
    16: 44,  # AES 128
    24: 52,  # AES 192
    32: 60   # AES 256
}


def _key_expansion(key_cols, col_in=0, col_out=None):
    """Compute AES key key expansion for any AES mode, given any known key bytes in the key schedule.

    Handle AES-128, AES-192, AES-256 modes, respectively 16, 24 and 32 bytes vector key length, given as numpy array input.
    Key expansion is computed from column index col_in to column index col_out, forward (resp. backward) if col_in < col_out (resp. col_in > col_out).

    Args:
        key_cols (numpy.ndarray): numpy byte array (dtype uint8). Last dimension must be 16, 24, or 32 long, for resp. AES 128, 192 and 256 mode.
        col_in (int, default=0): column index of the first key column provided. col_in should be between 0 and resp. 44 (52, 60) for 16 (24, 32) key length.
        col_out (int, default=None): column index of the last key column to expand. col_out should be between 0 and
            resp. 44 (52, 60) for 16 (24, 32) key length. If not provided, col_out is set to the maximum possible value.
    Returns:
        (numpy.ndarray): numpy byte array containing all expanded key bytes, with shape (number of keys, number of bytes).

    """
    
    _is_bytes_of_len(key_cols, length=[16, 24, 32])
    bytes_key_length = key_cols.shape[-1]
    number_of_keys = key_cols.shape[0] if key_cols.shape[:-1] else 1
    cols_in = int(bytes_key_length / 4)

    max_col_out = _cols_out[bytes_key_length]
    col_out = max_col_out if col_out is None else col_out

    if col_in < 0 or col_out < 0:
        raise ValueError(f'col_in and col_out must be greater than 0, not (resp.) {col_in} and {col_out}.')
    if col_out > max_col_out:
        raise ValueError(f'col_out should be lesser than {max_col_out}, not {col_out}.')

    if col_in < col_out:
        return _expand_forward(key_cols, bytes_key_length, cols_in, number_of_keys, col_in, col_out)
    else:
        return _expand_backward(key_cols, bytes_key_length, cols_in, number_of_keys, col_in, col_out)


def _expand_forward(key_cols, bytes_key_length, cols_in, number_of_keys, col_in, col_out):

    cols_range = range(col_in, col_out)
    n_cols = col_out - col_in

    key = key_cols.reshape((-1, cols_in, 4))
    expanded_key = _np.empty((number_of_keys, col_out, 4), dtype=_np.uint8)
    final_shape = (number_of_keys, (n_cols * 4))
    for index, col in enumerate(cols_range):
        if index < cols_in:
            expanded_key[:, col] = key[:, index, :]
        elif col % cols_in == 0:
            expanded_key[:, col] = SBOX[_np.roll(expanded_key[:, col - 1], shift=-1, axis=-1)]
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col], RCON[int(col / cols_in) - 1])
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col], expanded_key[:, col - cols_in])
        elif bytes_key_length == 32 and col % 4 == 0:
            expanded_key[:, col] = _np.bitwise_xor(
                SBOX[expanded_key[:, col - 1]],
                expanded_key[:, col - cols_in]
            )
        else:
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col - 1], expanded_key[:, col - cols_in])
    return expanded_key[:, col_in:].reshape(final_shape)


def _expand_backward(key_cols, bytes_key_length, cols_in, number_of_keys, col_in, col_out):
    cols_range = range(col_in + cols_in - 1, col_out - 1, -1)
    n_cols = col_in - col_out + cols_in

    key = key_cols.reshape((-1, cols_in, 4))
    expanded_key = _np.empty((number_of_keys, col_in + cols_in, 4), dtype=_np.uint8)
    final_shape = (number_of_keys, (n_cols * 4))
    for index, col in enumerate(cols_range):
        if index < cols_in:
            expanded_key[:, col] = key[:, cols_in - index - 1, :]
        elif col % cols_in == 0:
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col + cols_in], SBOX[_np.roll(expanded_key[:, col + cols_in - 1], shift=-1, axis=-1)])
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col], RCON[int(col / cols_in)])
        elif bytes_key_length == 32 and col % 4 == 0:
            expanded_key[:, col] = _np.bitwise_xor(
                SBOX[expanded_key[:, col + cols_in - 1]],
                expanded_key[:, col + cols_in])
        else:
            expanded_key[:, col] = _np.bitwise_xor(expanded_key[:, col + cols_in], expanded_key[:, col + cols_in - 1])
    return expanded_key[:, col_out:].reshape(final_shape)