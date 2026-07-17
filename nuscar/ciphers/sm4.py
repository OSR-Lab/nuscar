# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from typing import Callable
import nuscar
from makefun import with_signature
from inspect import Signature, Parameter
import enum


class Steps(enum.IntEnum):
    """Enumeration for the four SM4 round steps."""
    AK = 0
    SBox = 1
    L = 2
    XORX0 = 3
    ROUT = 4
    FinalX = 5


SBOX = _np.array([
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48],
    dtype=_np.uint8)

CK = _np.array([
    0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
    0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
    0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
    0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
    0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
    0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
    0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
    0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279],
    dtype=_np.uint32)

FK = _np.array([
    0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc],
    dtype=_np.uint32)


def sub_bytes(state):
    return SBOX[state]


def _sub_bytes_word(input_data):
    input_data = _np.array(input_data)
    return SBOX[input_data.view(dtype=_np.uint8)].view(dtype=_np.uint32)


def l(input: _np.ndarray):
    """
    sm4 L operation.
    """
    if input.shape[-1] != 4:
        raise ValueError("L operation must be applied to a 4-byte word")
    input = input.reshape(-1, 4)
    return _n_rust.sm4_opl_r(nuscar._global_pool, input)


def _bytes_to_word(input_data):
    """
    convert numpy table with 4 dtype uint8 to uint32, left byte most significant
    Parameter:
        input_data: multi-byte dimension numpy table, cell number should be multiple of 4
    Output:
        result of converted table
    """
    input_data = input_data.reshape((-1, 4)).astype(dtype=_np.uint32)
    # res = _np.empty(input_data.shape[0], dtype=_np.uint32)
    res = (input_data[:, 0] << 24) ^ (input_data[:, 1] << 16) ^ (
        input_data[:, 2] << 8) ^ input_data[:, 3]
    return res


def _word_to_bytes(input_data):
    """
    convert numpy table with dtype uint32 to  4 uint8, left byte most significant
    Parameter:
        input_data: multi-byte dimension numpy table
    Output:
        result of converted table
    """
    input_data = input_data.reshape(-1).astype(dtype=_np.uint32)
    res = _np.empty([input_data.shape[0], 4], dtype=_np.uint8)
    res[:, 0] = input_data >> 24
    res[:, 1] = (input_data >> 16) & 0xFF
    res[:, 2] = (input_data >> 8) & 0xFF
    res[:, 3] = input_data & 0xFF
    return res


def _circ_shift_32(input_data, n):
    """
    circle left shift operation on 4-byte word
    Parameter:
        input_data: multi-byte dimension numpy table, last dimension should be 1
        n: number of bit to shift, n should not larger than 32
    Output:
        result of circle left shift operation
    """
    if n > 32:
        raise ValueError(
            'Wrong shift bit number, n should not larger than 32, not ' + str(n))
    if not input_data.dtype == _np.uint32:
        raise ValueError('Wrong dtype, uint32 expected, but ' +
                         str(input_data.dtype) + 'given')
    input_data = input_data.reshape(-1)
    return _np.bitwise_xor(_np.left_shift(input_data, n), _np.right_shift(input_data, 32 - n))


def l_key(input_data):
    """
        SM4 L' operation on 4-byte word, used in key schedule
        Parameter:
            input_data: multi-byte dimension numpy table, last dimension should be 4 bytes vectors
        Output:
            result of L operation
        """
    input_data = input_data.reshape(-1)
    return _np.bitwise_xor(input_data, _np.bitwise_xor(_circ_shift_32(input_data, 13), _circ_shift_32(input_data, 23)))


def key_schedule(original_key):
    """
    SM4 key schedule operation
    Parameter:
        original: multi-byte dimension numpy table, last dimension should be 16 numpy bytes array
    Output:
        scheduled_key: numpy table with dtype uint32, with last dimension being 32 (32 rounds)
    """
    if original_key.shape[-1] != 16:
        raise ValueError('Wrong size of SM4 key, 16 expected but ' +
                         str(original_key.shape(-1)) + ' bytes obtained')
    original_key = original_key.reshape((-1, 16))
    output_key = _np.empty([original_key.shape[0], 32], dtype=_np.uint32)

    input_key = _bytes_to_word(
        original_key).reshape((-1, 4)) ^ FK
    for i in range(32):
        output_key[:, i] = input_key[:, 0] ^ l_key(_sub_bytes_word(
            input_key[:, 1] ^ input_key[:, 2] ^ input_key[:, 3] ^ CK[i]))
        input_key = _np.roll(input_key, -1, axis=1)
        input_key[:, 3] = output_key[:, i]
    return output_key


def inv_key_schedule(round_key, round_start=0):
    """
    SM4 inv key schedule operation
    Parameter:
        original: multi-byte dimension numpy table, last dimension should be 4 numpy bytes array , second last dimension should be 4(4 continuous round key)
    Output:
        master key
    """
    if round_key.shape[-1] != 4 or round_key.shape[-2] != 4:
        raise ValueError("Wrong size of round key. Last dimension should be 4 numpy bytes array , second last dimension should "
                         "be 4(4 continuous round key)")
    if round_start > 28 or round_start < 0:
        raise ValueError(
            "round_start should not be negative and should not lager than 28")

    input_key = _bytes_to_word(round_key).reshape((-1, 4))
    input_key = _np.fliplr(input_key)
    for i in range(round_start + 4):
        tmp = input_key[:, 0] ^ l_key(
            _sub_bytes_word(input_key[:, 1] ^ input_key[:, 2] ^ input_key[:, 3] ^ CK[round_start + 3 - i]))
        input_key = _np.roll(input_key, -1, axis=1)
        input_key[:, 3] = tmp

    input_key = _np.fliplr(input_key)
    input_key = input_key ^ FK
    return _word_to_bytes(input_key).reshape(-1, 16)


def encrypt(plaintext: _np.ndarray, key: _np.ndarray, stop_round=31, stop_step: Steps = Steps.FinalX) -> _np.ndarray:
    """sm4 encrypt plaintext with given key, may return intermediate state at given stop_round and stop_step.

    Args:
        plaintext (ndarray): 1-D/2-D plaintext.
        key (ndarray): 1-D/2-D key.
        stop_round (int, optional): which round to stop, start from round 0. Defaults to 31. 
        stop_step (Steps, optional): which step to stop. Defaults to Steps.FinalX. Only round 31 have FinalX step.
    """

    if stop_round != 31 and stop_step == Steps.FinalX:
        raise ValueError(f"SM4 round %d does not have %s operation" %
                         (stop_round, str(stop_step)))
    if stop_round > 31:
        raise ValueError("SM4 round cannot exceed 31")

    nb_words = plaintext.shape[-1]
    plaintext = plaintext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.sm4_encrypt_step_r(
            nuscar._global_pool, plaintext, key, stop_round, stop_step.value)
    elif (len(key.shape) == 1):
        return _n_rust.sm4_encrypt_fix_key_step_r(
            nuscar._global_pool, plaintext, key, stop_round, stop_step.value)
    else:
        raise ValueError(f"key shape cannot be %s" % str(key.shape))

def decrypt(ciphertext: _np.ndarray, key: _np.ndarray, stop_round=31, stop_step: Steps = Steps.FinalX) -> _np.ndarray:
    """sm4 decrypt ciphertext with given key, may return intermediate state at given stop_round and stop_step.

    Args:
        ciphertext (ndarray): 1-D/2-D plaintext.
        key (ndarray): 1-D/2-D key.
        stop_round (int, optional): which round to stop, start from round 0. Defaults to 31.
        stop_step (Steps, optional): which step to stop. Defaults to Steps.FinalX. Only round 31 have FinalX step.
    """

    if stop_round != 31 and stop_step == Steps.FinalX:
        raise ValueError(f"SM4 round %d does not have %s operation" %
                         (stop_round, str(stop_step)))
    if stop_round > 31:
        raise ValueError("SM4 round cannot exceed 31")

    nb_words = ciphertext.shape[-1]
    ciphertext = ciphertext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.sm4_decrypt_step_r(
            nuscar._global_pool, ciphertext, key, stop_round, stop_step.value)
    elif (len(key.shape) == 1):
        return _n_rust.sm4_decrypt_fix_key_step_r(
            nuscar._global_pool, ciphertext, key, stop_round, stop_step.value)
    else:
        raise ValueError(f"key shape cannot be %s" % str(key.shape))


def attack_sbox_hw(meta_name='plaintext', rk=None, pos=list(range(4)), guesses=list(range(256))) -> Callable:
    """generate a selection function targeting hamming weight of sbox out.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        rk (ndarray): 2-D ndarray. will attack round len(rk)+1, if rk is None, attack the first round key.
        pos (list, optional): key positions to attack. Note only 4 bytes key in one sm4 round, defaults to list(range(4)).
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
        plaintext = plaintext.reshape(-1, plaintext.shape[-1])
        if rk is None:
            return _n_rust.sm4_attack_first_sbox_hw_with_guess_r(nuscar._global_pool, plaintext, kwargs["guesses"], pos)
        else:
            return _n_rust.sm4_attack_sbox_hw_with_guess_r(nuscar._global_pool, plaintext, rk, kwargs["guesses"], pos)
    return func_impl


def attack_sbox_bit(meta_name='plaintext', rk=None, pos=list(range(4)), bit_pos=0, guesses=list(range(256)), ) -> Callable:
    """generate a selection function targeting one bit of sbox out.

    Args:
        meta_name (str, optional): meta name in container. Defaults to 'plaintext'.
        rk (ndarray): 2-D ndarray. will attack round len(rk)+1, if rk is None, attack the first round key.
        pos (list, optional): key positions to attack. Note only 4 bytes key in one sm4 round, defaults to list(range(4)).
        bit_pos: which bit to attack.
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
        plaintext = plaintext.reshape(-1, plaintext.shape[-1])
        if rk is None:
            return _n_rust.sm4_attack_first_sbox_bit_with_guess_r(nuscar._global_pool, plaintext, kwargs["guesses"], pos, bit_pos)
        else:
            return _n_rust.sm4_attack_sbox_bit_with_guess_r(nuscar._global_pool, plaintext, rk, kwargs["guesses"], pos, bit_pos)
    return func_impl
