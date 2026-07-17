# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import enum
import numpy as _np
import nuscar
import nuscar.nuscar_rust as _n_rust
from typing import Callable
from makefun import with_signature
from inspect import Signature, Parameter

SBOXES = _np.array([
    [0xE, 0x0, 0x4, 0xF, 0xD, 0x7, 0x1, 0x4, 0x2, 0xE, 0xF, 0x2, 0xB, 0xD, 0x8, 0x1, 0x3, 0xA, 0xA, 0x6, 0x6,
        0xC, 0xC, 0xB, 0x5, 0x9, 0x9, 0x5, 0x0, 0x3, 0x7, 0x8, 0x4, 0xF, 0x1, 0xC, 0xE, 0x8, 0x8, 0x2, 0xD, 0x4,
        0x6, 0x9, 0x2, 0x1, 0xB, 0x7, 0xF, 0x5, 0xC, 0xB, 0x9, 0x3, 0x7, 0xE, 0x3, 0xA, 0xA, 0x0, 0x5, 0x6, 0x0,
        0xD],
    [0xF, 0x3, 0x1, 0xD, 0x8, 0x4, 0xE, 0x7, 0x6, 0xF, 0xB, 0x2, 0x3, 0x8, 0x4, 0xE, 0x9, 0xC, 0x7, 0x0, 0x2,
        0x1, 0xD, 0xA, 0xC, 0x6, 0x0, 0x9, 0x5, 0xB, 0xA, 0x5, 0x0, 0xD, 0xE, 0x8, 0x7, 0xA, 0xB, 0x1, 0xA, 0x3,
        0x4, 0xF, 0xD, 0x4, 0x1, 0x2, 0x5, 0xB, 0x8, 0x6, 0xC, 0x7, 0x6, 0xC, 0x9, 0x0, 0x3, 0x5, 0x2, 0xE, 0xF,
        0x9],
    [0xA, 0xD, 0x0, 0x7, 0x9, 0x0, 0xE, 0x9, 0x6, 0x3, 0x3, 0x4, 0xF, 0x6, 0x5, 0xA, 0x1, 0x2, 0xD, 0x8, 0xC,
        0x5, 0x7, 0xE, 0xB, 0xC, 0x4, 0xB, 0x2, 0xF, 0x8, 0x1, 0xD, 0x1, 0x6, 0xA, 0x4, 0xD, 0x9, 0x0, 0x8, 0x6,
        0xF, 0x9, 0x3, 0x8, 0x0, 0x7, 0xB, 0x4, 0x1, 0xF, 0x2, 0xE, 0xC, 0x3, 0x5, 0xB, 0xA, 0x5, 0xE, 0x2, 0x7,
        0xC],
    [0x7, 0xD, 0xD, 0x8, 0xE, 0xB, 0x3, 0x5, 0x0, 0x6, 0x6, 0xF, 0x9, 0x0, 0xA, 0x3, 0x1, 0x4, 0x2, 0x7, 0x8,
        0x2, 0x5, 0xC, 0xB, 0x1, 0xC, 0xA, 0x4, 0xE, 0xF, 0x9, 0xA, 0x3, 0x6, 0xF, 0x9, 0x0, 0x0, 0x6, 0xC, 0xA,
        0xB, 0x1, 0x7, 0xD, 0xD, 0x8, 0xF, 0x9, 0x1, 0x4, 0x3, 0x5, 0xE, 0xB, 0x5, 0xC, 0x2, 0x7, 0x8, 0x2, 0x4,
        0xE],
    [0x2, 0xE, 0xC, 0xB, 0x4, 0x2, 0x1, 0xC, 0x7, 0x4, 0xA, 0x7, 0xB, 0xD, 0x6, 0x1, 0x8, 0x5, 0x5, 0x0, 0x3,
        0xF, 0xF, 0xA, 0xD, 0x3, 0x0, 0x9, 0xE, 0x8, 0x9, 0x6, 0x4, 0xB, 0x2, 0x8, 0x1, 0xC, 0xB, 0x7, 0xA, 0x1,
        0xD, 0xE, 0x7, 0x2, 0x8, 0xD, 0xF, 0x6, 0x9, 0xF, 0xC, 0x0, 0x5, 0x9, 0x6, 0xA, 0x3, 0x4, 0x0, 0x5, 0xE,
        0x3],
    [0xC, 0xA, 0x1, 0xF, 0xA, 0x4, 0xF, 0x2, 0x9, 0x7, 0x2, 0xC, 0x6, 0x9, 0x8, 0x5, 0x0, 0x6, 0xD, 0x1, 0x3,
        0xD, 0x4, 0xE, 0xE, 0x0, 0x7, 0xB, 0x5, 0x3, 0xB, 0x8, 0x9, 0x4, 0xE, 0x3, 0xF, 0x2, 0x5, 0xC, 0x2, 0x9,
        0x8, 0x5, 0xC, 0xF, 0x3, 0xA, 0x7, 0xB, 0x0, 0xE, 0x4, 0x1, 0xA, 0x7, 0x1, 0x6, 0xD, 0x0, 0xB, 0x8, 0x6,
        0xD],
    [0x4, 0xD, 0xB, 0x0, 0x2, 0xB, 0xE, 0x7, 0xF, 0x4, 0x0, 0x9, 0x8, 0x1, 0xD, 0xA, 0x3, 0xE, 0xC, 0x3, 0x9,
        0x5, 0x7, 0xC, 0x5, 0x2, 0xA, 0xF, 0x6, 0x8, 0x1, 0x6, 0x1, 0x6, 0x4, 0xB, 0xB, 0xD, 0xD, 0x8, 0xC, 0x1,
        0x3, 0x4, 0x7, 0xA, 0xE, 0x7, 0xA, 0x9, 0xF, 0x5, 0x6, 0x0, 0x8, 0xF, 0x0, 0xE, 0x5, 0x2, 0x9, 0x3, 0x2,
        0xC],
    [0xD, 0x1, 0x2, 0xF, 0x8, 0xD, 0x4, 0x8, 0x6, 0xA, 0xF, 0x3, 0xB, 0x7, 0x1, 0x4, 0xA, 0xC, 0x9, 0x5, 0x3,
        0x6, 0xE, 0xB, 0x5, 0x0, 0x0, 0xE, 0xC, 0x9, 0x7, 0x2, 0x7, 0x2, 0xB, 0x1, 0x4, 0xE, 0x1, 0x7, 0x9, 0x4,
        0xC, 0xA, 0xE, 0x8, 0x2, 0xD, 0x0, 0xF, 0x6, 0xC, 0xA, 0x9, 0xD, 0x0, 0xF, 0x3, 0x3, 0x5, 0x5, 0x6, 0x8,
        0xB]
], dtype=_np.uint8)


PC1 = [57, 49, 41, 33, 25, 17, 9,
       1, 58, 50, 42, 34, 26, 18,
       10, 2, 59, 51, 43, 35, 27,
       19, 11, 3, 60, 52, 44, 36,
       63, 55, 47, 39, 31, 23, 15,
       7, 62, 54, 46, 38, 30, 22,
       14, 6, 61, 53, 45, 37, 29,
       21, 13, 5, 28, 20, 12, 4]


PC2 = [14, 17, 11, 24, 1, 5,
       3, 28, 15, 6, 21, 10,
       23, 19, 12, 4, 26, 8,
       16, 7, 27, 20, 13, 2,
       41, 52, 31, 37, 47, 55,
       30, 40, 51, 45, 33, 48,
       44, 49, 39, 56, 34, 53,
       46, 42, 50, 36, 29, 32]


DES_ROUNDS = 16


class Steps(enum.IntEnum):
    """Enumeration for the DES round steps."""

    INITIAL_PERMUTATION = 0  # Resulting in LR (Left-Right)
    EXPANSIVE_PERMUTATION = 1  # Expansion on R part
    ADD_ROUND_KEY = 2
    SBOXES = 3
    PERMUTATION_P = 4
    XOR_WITH_SAVED_LEFT_RIGHT = 5  # Thus we obtain new Right-Left
    PERMUTE_RIGHT_LEFT = 6  # Thus we obtain Left-Right for next round
    INV_PERMUTATION_P_RIGHT = 7  # Inverted permutation P of Ri
    INV_PERMUTATION_P_DELTA_RIGHT = 8  # Inverted permutation P of (Ri xor Ri-1)
    FINAL_PERMUTATION = 9  # The final cipher

def _is_bytes_array(array):
    # Note: Integer arrays cannot contain np.nan or np.inf
    if not isinstance(array, _np.ndarray):
        raise TypeError(f'array should be a Numpy ndarray instance, not {type(array)}.')
    if array.dtype == _np.uint8:
        return True
    if array.dtype.kind not in 'ui':
        raise ValueError(f'array should be an integer array, not {array.dtype}.')
    if array.dtype.kind == 'i' and _np.min(array) < 0:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but lowest value {_np.min(array)} found.')
    if array.dtype != _np.int8 and _np.max(array) > 255:
        raise ValueError(f'array should be a bytes array, i.e with values in [0, 255], but highest value {_np.max(array)} found.')
    return True

def _is_bytes_of_len(state, length=[8]):
    _is_bytes_array(state)
    if state.shape[-1] not in length:
        raise ValueError(f'state last dimension should be in {length}, not {state.shape[-1]}.')
    return True


def key_schedule(key, interrupt_after_round=15):
    """Compute DES key schedule.

    Args:
        key (numpy.ndarray): numpy byte array (dtype uint8), with last dimension 8 bytes long. The key to use as input.
        interrupt_after_round (int): last round to include (0-15). Defaults to 15 (all rounds).

    Returns:
        (numpy.ndarray): numpy byte array containing all round keys up to interrupt_after_round, with shape (number of keys, number of rounds, 8),
            or (number of rounds, 8) if only one key has been provided. All the generated round keys are 6bits words.

    Examples:
        import numpy as np
        key = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF], dtype=np.uint8)
        schedule = key_schedule(key)

    Raises:
        TypeError: For key not being a numpy array, or for interrupt_after_round not being an int.
        ValueError: For key not being uint8 type, or containing keys longer than 8 bytes. Or for interrupt_after_round not being between 0 and 15.

    """
    _is_bytes_of_len(key)
    if not isinstance(interrupt_after_round, int):
        raise TypeError(f"Wrong argument type for interrupt_after_round, got {type(interrupt_after_round)} instead of int.")
    if interrupt_after_round < 0 or interrupt_after_round > 15:
        raise ValueError(f"Wrong DES key round number, must be between 0 and 15 and not {interrupt_after_round}.")

    dimensions = key.shape[:-1]
    data = key.reshape((-1, 8))
    full_schedule = _n_rust.des_key_schedule_r(nuscar._global_pool, data)
    truncated = full_schedule[:, :interrupt_after_round + 1, :]
    if dimensions:
        return truncated.reshape(dimensions + (interrupt_after_round + 1, 8))
    else:
        return truncated.reshape((interrupt_after_round + 1, 8))


def get_master_key(round_key, nb_round, plaintext, expected_ciphertext):
    """Retrieve the DES master key from a key schedule round key.

    Compute the 256 possible master keys from a round key. Then, from the 256 possible masters keys, compute a DES with the provided plaintext,
    and check if the expected_ciphertext is found. If so, the correct key is discovered and will be returned.

    Args:
        round_key (numpy.ndarray): the round key to analyze. 8x6bit word.
        nb_round (int): the number of the round corresponding to the round key.
        plaintext (numpy.ndarray): a 8 byte input plaintext.
        expected_ciphertext (numpy.ndarray): a 8 byte output message computed from the key to discover.

    Returns:
        (numpy.ndarray) the found master key, or None if not found.

    """
    _is_bytes_of_len(round_key, length=[8])
    _is_bytes_of_len(plaintext, length=[8])
    _is_bytes_of_len(expected_ciphertext, length=[8])
    if not (round_key < 64).all():
        raise ValueError('round_key should be a 8x6bit array, but it contains at least one value coded on more than 6 bits (> 63).')
    if not isinstance(nb_round, int):
        raise TypeError(f'nb_round must be an integer value, not a value of type {type(nb_round)}.')
    if nb_round < 0 or nb_round > 15:
        raise ValueError(f'nb_round must be between 0 and 15, not {nb_round}.')

    guess_keys = _find_possible_keys(round_key, nb_round)
    for guess in guess_keys:
        computed_ciphertext = encrypt(plaintext, guess).flatten()
        if _np.array_equal(expected_ciphertext.flatten(), computed_ciphertext):
            return guess
    return None


def _find_possible_keys(round_key, nb_round):
    # This is ci_di before performing PC-2
    # 255 value means the bit value is unknown
    ci_di = _np.array([0, 0, 0, 0, 0, 0,
                       0, 0, 255, 0, 0, 0,
                       0, 0, 0, 0, 0, 255,
                       0, 0, 0, 255, 0, 0, 255,
                       0, 0, 0, 0, 0, 0, 0,
                       0, 0, 255, 0, 0, 255,
                       0, 0, 0, 0, 255, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 255, 0, 0], dtype=_np.uint8)
    # We remove PC-2
    for index in range(48):
        word = int(index / 6)
        bit = 1 << (5 - index % 6)
        if round_key[word] & bit != 0:
            ci_di[PC2[index] - 1] = 1
    # We split in Ci | Di
    ci_di = ci_di.reshape(2, 28)
    # We remove the left shits according to the round number
    nb_shift = [1, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 28]
    ci_di = _np.roll(ci_di, + nb_shift[nb_round], 1)
    # We come back as a 56 bit array before PC-1
    ci_di = ci_di.reshape(56)
    # We remove PC-1
    master_key = [0] * 64
    for index in range(len(PC1)):
        master_key[PC1[index] - 1] = ci_di[index]
    # We can now compute the 256 possible keys, and return them in an array
    guessed_keys = _convert_hypothesis_bits_into_keys(master_key)
    return _np.array([[(guess >> (8 * hit)) & 0xFF for hit in range(7, -1, -1)] for guess in guessed_keys], dtype=_np.uint8)


def _convert_hypothesis_bits_into_keys(array):
    len_array = len(array)
    if len_array == 1:
        return [array[0]]
    else:
        bit = array[0]
        keys_for_value_0 = _convert_hypothesis_bits_into_keys(array[1:])
        if not bit:
            return keys_for_value_0
        keys_for_value_1 = [hit + (1 << (len_array - 1)) for hit in keys_for_value_0]
        if bit == 255:
            # If the bit is unknown, we have to return both the values for bit 0 and 1
            return keys_for_value_1 + keys_for_value_0
        else:
            return keys_for_value_1


def initial_permutation(state):
    """Compute DES initial permutation (IP) operation resulting on a 8 bytes words state.

    This operation outputs the L0R0 value.

    Args:
        state (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES initial permutation result with same dimensions as input state.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _n_rust.des_initial_permutation_r(nuscar._global_pool, data)
    return out.reshape(dimensions)


def expansive_permutation(state):
    """Compute DES expansive permutation (EP) operation resulting on a 8x6bit words state.

    Args:
        state (numpy.ndarray): a uint8 array of 4 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES expansive permutation result with 8x6bit dimension words.

    """
    _is_bytes_of_len(state, length=[4])
    dimensions = state.shape
    data = state.reshape((-1, 4))
    out = _n_rust.des_expansive_permutation_r(nuscar._global_pool, data)
    return out.reshape(dimensions[:-1] + (8,))


def add_round_key(state, keys):
    """Compute DES xor operation between a words state and a round keys array, both 8x6bit.

    Depending on the shapes of state and keys, the result can be:

        - one state added to one key if state is (8,) and keys is (8,)
        - one state added to n keys if state is (8,) and keys is (n, 8)
        - n states added to 1 key if state is (n, 8) and keys is (8)
        - states added to keys, combined by pairs if state is (n, 8) and keys is (n, 8).

    In every other case, a ValueError will be raised.

    Args:
        state (numpy.ndarray): a uint8 array of 8x6bit words as last dimension.
        keys (numpy.ndarray): a uint8 array of 8x6bit round keys as last dimension.

    Returns:
        (numpy.ndarray) xor result between state and keys.

    """
    _is_bytes_of_len(state)
    _is_bytes_of_len(keys)
    return _np.bitwise_xor(state, keys)


def sboxes(state):
    """Compute DES SBOXes operation resulting on a 8x4bits words state.

    Args:
        state (numpy.ndarray): a uint8 array of 8x6bits words as last dimension.

    Returns:
        (numpy.ndarray) DES SBOXes operation result with 8x4bits words.

    """
    _is_bytes_of_len(state)
    out = _np.empty_like(state)
    for i in range(8):
        out[..., i] = SBOXES[i][state[..., i]]
    return out


def permutation_p(state):
    """Compute DES permutation P (PP) operation resulting on a 4 bytes words state.

    Args:
        state (numpy.ndarray): a uint8 array of 8x4bits words as last dimension.

    Returns:
        (numpy.ndarray) DES permutation P result with 4 bytes dimension words.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _n_rust.des_permutation_p_r(nuscar._global_pool, data)
    return out.reshape(dimensions[:-1] + (4,))


def inv_permutation_p(state):
    """Compute inverse of DES permutation P (PP) operation, resulting on a 8x4bits words state.

    Args:
        state (numpy.ndarray): a uint8 array of 4 bytes words as last dimension.

    Returns:
        (numpy.ndarray) Inverse of DES permutation P result with 8x4bits dimension words.

    """
    _is_bytes_of_len(state, length=[4])
    dimensions = state.shape
    data = state.reshape((-1, 4))
    out = _n_rust.des_inv_permutation_p_r(nuscar._global_pool, data)
    return out.reshape(dimensions[:-1] + (8,))


def final_permutation(state):
    """Compute DES final permutation (FP) operation resulting on a 8 bytes words state.

    This operation outputs the R16L16 value, ie: the final ciphertext.

    Args:
        state (numpy.ndarray): a uint8 array of 8 bytes words as last dimension.

    Returns:
        (numpy.ndarray) DES final permutation result with same dimensions as input state.

    """
    _is_bytes_of_len(state)
    dimensions = state.shape
    data = state.reshape((-1, 8))
    out = _n_rust.des_final_permutation_r(nuscar._global_pool, data)
    return out.reshape(dimensions)


def encrypt(plaintext, key, at_round=None, after_step=Steps.FINAL_PERMUTATION):
    """DES encrypt plaintext with given key, may return intermediate state at given at_round and after_step.

    Args:
        plaintext (numpy.ndarray): 1-D/2-D plaintext, last dimension 8 bytes.
        key (numpy.ndarray): 1-D/2-D key, last dimension 8 bytes.
        at_round (int, optional): which round to stop. Defaults to 15.
        after_step (Steps, optional): which step to stop. Defaults to Steps.FINAL_PERMUTATION.
    """
    if at_round is None:
        at_round = DES_ROUNDS - 1
    if not isinstance(at_round, (int, _np.integer)) or at_round < 0 or at_round >= DES_ROUNDS:
        raise ValueError(f"at_round must be between 0 and {DES_ROUNDS - 1}, not {at_round}.")

    nb_words = plaintext.shape[-1]
    plaintext = plaintext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.des_encrypt_step_r(
            nuscar._global_pool, plaintext, key, int(at_round), int(after_step))
    elif len(key.shape) == 1:
        return _n_rust.des_encrypt_step_fix_key_r(
            nuscar._global_pool, plaintext, key, int(at_round), int(after_step))
    else:
        raise ValueError(f"key shape error: {key.shape}")


def decrypt(ciphertext, key, at_round=None, after_step=Steps.FINAL_PERMUTATION):
    """DES decrypt ciphertext with given key, may return intermediate state at given at_round and after_step.

    Args:
        ciphertext (numpy.ndarray): 1-D/2-D ciphertext, last dimension 8 bytes.
        key (numpy.ndarray): 1-D/2-D key, last dimension 8 bytes.
        at_round (int, optional): which round to stop. Defaults to 15.
        after_step (Steps, optional): which step to stop. Defaults to Steps.FINAL_PERMUTATION.
    """
    if at_round is None:
        at_round = DES_ROUNDS - 1
    if not isinstance(at_round, (int, _np.integer)) or at_round < 0 or at_round >= DES_ROUNDS:
        raise ValueError(f"at_round must be between 0 and {DES_ROUNDS - 1}, not {at_round}.")

    nb_words = ciphertext.shape[-1]
    ciphertext = ciphertext.reshape(-1, nb_words)

    if len(key.shape) == 2:
        return _n_rust.des_decrypt_step_r(
            nuscar._global_pool, ciphertext, key, int(at_round), int(after_step))
    elif len(key.shape) == 1:
        return _n_rust.des_decrypt_step_fix_key_r(
            nuscar._global_pool, ciphertext, key, int(at_round), int(after_step))
    else:
        raise ValueError(f"key shape error: {key.shape}")



def _make_attack(rust_fn, default_meta='plaintext'):
    """Factory function for generating DES attack selection functions."""
    def attack_factory(meta_name=default_meta, guesses=list(range(64))) -> Callable:
        parameters = [Parameter(meta_name, kind=Parameter.POSITIONAL_OR_KEYWORD),
                      Parameter("guesses", kind=Parameter.POSITIONAL_OR_KEYWORD, default=guesses)]
        func_sig = Signature(parameters)

        @with_signature(func_sig, func_name="func")
        def func_impl(**kwargs):
            data = kwargs[meta_name]
            if data.ndim == 1:
                data = _np.expand_dims(data, axis=0)
            return rust_fn(nuscar._global_pool, data, kwargs["guesses"])

        return func_impl
    return attack_factory


# First-round (encrypt) attack functions
attack_first_addRk_hw    = _make_attack(_n_rust.des_attack_first_addRk_hw_with_guess_r)
attack_first_addRk_value = _make_attack(_n_rust.des_attack_first_addRk_value_with_guess_r)
attack_first_sbox_hw     = _make_attack(_n_rust.des_attack_first_sbox_hw_with_guess_r)
attack_first_sbox_value  = _make_attack(_n_rust.des_attack_first_sbox_value_with_guess_r)
attack_first_round_hw    = _make_attack(_n_rust.des_attack_first_round_hw_with_guess_r)
attack_first_round_value = _make_attack(_n_rust.des_attack_first_round_value_with_guess_r)
attack_delta_first_round_hw    = _make_attack(_n_rust.des_attack_delta_first_round_hw_with_guess_r)
attack_delta_first_round_value = _make_attack(_n_rust.des_attack_delta_first_round_value_with_guess_r)

# Last-round (decrypt) attack functions
attack_last_addRk_hw    = _make_attack(_n_rust.des_attack_last_addRk_hw_with_guess_r, 'ciphertext')
attack_last_addRk_value = _make_attack(_n_rust.des_attack_last_addRk_value_with_guess_r, 'ciphertext')
attack_last_sbox_hw     = _make_attack(_n_rust.des_attack_last_sbox_hw_with_guess_r, 'ciphertext')
attack_last_sbox_value  = _make_attack(_n_rust.des_attack_last_sbox_value_with_guess_r, 'ciphertext')
attack_last_round_hw    = _make_attack(_n_rust.des_attack_last_round_hw_with_guess_r, 'ciphertext')
attack_last_round_value = _make_attack(_n_rust.des_attack_last_round_value_with_guess_r, 'ciphertext')
attack_delta_last_round_hw    = _make_attack(_n_rust.des_attack_delta_last_round_hw_with_guess_r, 'ciphertext')
attack_delta_last_round_value = _make_attack(_n_rust.des_attack_delta_last_round_value_with_guess_r, 'ciphertext')


def _make_compute(cipher_fn, step, apply_hw):
    """Factory function for generating DES compute selection functions."""
    if apply_hw:
        def compute(data, key):
            res = cipher_fn(data, key, 0, step)
            return nuscar.leakmodel.leakage_model_hw(res)
    else:
        def compute(data, key):
            return cipher_fn(data, key, 0, step)
    return compute


# First-round compute functions
compute_first_addRk_hw    = _make_compute(encrypt, Steps.ADD_ROUND_KEY, True)
compute_first_addRk_value = _make_compute(encrypt, Steps.ADD_ROUND_KEY, False)
compute_first_sbox_hw     = _make_compute(encrypt, Steps.SBOXES, True)
compute_first_sbox_value  = _make_compute(encrypt, Steps.SBOXES, False)
compute_first_round_hw    = _make_compute(encrypt, Steps.INV_PERMUTATION_P_RIGHT, True)
compute_first_round_value = _make_compute(encrypt, Steps.INV_PERMUTATION_P_RIGHT, False)
compute_delta_first_round_hw    = _make_compute(encrypt, Steps.INV_PERMUTATION_P_DELTA_RIGHT, True)
compute_delta_first_round_value = _make_compute(encrypt, Steps.INV_PERMUTATION_P_DELTA_RIGHT, False)

# Last-round compute functions
compute_last_addRk_hw    = _make_compute(decrypt, Steps.ADD_ROUND_KEY, True)
compute_last_addRk_value = _make_compute(decrypt, Steps.ADD_ROUND_KEY, False)
compute_last_sbox_hw     = _make_compute(decrypt, Steps.SBOXES, True)
compute_last_sbox_value  = _make_compute(decrypt, Steps.SBOXES, False)
compute_last_round_hw    = _make_compute(decrypt, Steps.INV_PERMUTATION_P_RIGHT, True)
compute_last_round_value = _make_compute(decrypt, Steps.INV_PERMUTATION_P_RIGHT, False)
compute_delta_last_round_hw    = _make_compute(decrypt, Steps.INV_PERMUTATION_P_DELTA_RIGHT, True)
compute_delta_last_round_value = _make_compute(decrypt, Steps.INV_PERMUTATION_P_DELTA_RIGHT, False)
