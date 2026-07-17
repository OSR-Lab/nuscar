# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
import nuscar


def leakage_model_hw(data: _np.ndarray, nb_words: int = 1) -> _np.ndarray:
    if not data.dtype == _np.uint8:
        raise ValueError("data type must be uint8")
    if data.shape[-1] % nb_words:
        raise ValueError(f"Data length %d is not divisible by nb_words=%d" % (data.shape[-1], nb_words))
    data_shape = list(data.shape)
    
    r = _n_rust.leak_hamming_weight_row_r(nuscar._global_pool, data.reshape(-1, nb_words))
    data_shape[-1] = data_shape[-1]//nb_words
    return r.reshape(data_shape)


def leakage_model_bit(data: _np.ndarray, bit_pos: int = 0) -> _np.ndarray:
    if not data.dtype == _np.uint8:
        raise ValueError("data type must be uint8")

    return _np.bitwise_and(_np.right_shift(data, bit_pos), 1)
