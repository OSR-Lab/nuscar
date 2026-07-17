# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.
import numpy as _np
import enum
import nuscar.nuscar_rust as _n_rust
import nuscar

class COMBINE_MODE(enum.IntEnum):
    """Enumeration high order combine mode."""

    FULL = 0
    SAME = 1

def get_precision(precision: str):
    if precision == 'f64':
        return _np.float64
    elif precision == 'f32':
        return _np.float32
    else:
        raise ValueError("precision must be 'f64' or 'f32'")

def _check_frames(frame1, frame2, mode):
    if isinstance(frame1, int):
        frame1 = [frame1]
    if isinstance(frame2, int):
        frame2 = [frame2]
    if frame2 is None:
        return frame1, frame1
    if mode == COMBINE_MODE.SAME:
        if len(frame1) != len(frame2):
            raise ValueError("In SAME mode, frame1 and frame2 must have equal lengths!")
    return frame1, frame2


def product(data: _np.ndarray, frame1, frame2=None, mode: COMBINE_MODE = COMBINE_MODE.FULL, precision='f64'):
    """High order combination with combine function product. e.g. x*y.

    Args:
        data (ndarray): 1-D/2-D input samples.
        frame1 (slice or list): first traces frame. 
        frame2 (slice or list): second traces frame. Defaults to None, reuse frame1.
        mode (str, optional): Defaults to 'full'.  
            In SAME mode, every sample of `frame1` will be combined with the corresponding sample in frame2. 
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x1·y1, x2·y2]
            In FULL mode, every sample of `frame1` is combined with full `frame2`.
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x0·y1, x0·y2, x1·y0, x1·y1, x1·y2, x2·y0, x2·y1, x2·y2]
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
    """
    dtype = get_precision(precision)
    frame1, frame2 = _check_frames(frame1, frame2, mode)
    nb_samples = data.shape[-1]
    data = data.reshape(-1, nb_samples)
    fs1 = data[:, frame1].astype(dtype)
    fs2 = data[:, frame2].astype(dtype)
    if mode == COMBINE_MODE.SAME:
        out = fs1 * fs2
    else:
        if precision == 'f64':
            out = _n_rust.combine_product_r(nuscar._global_pool, fs1, fs2)
        else:
            out = _n_rust.combine_product_r32(nuscar._global_pool, fs1, fs2)
    return _np.squeeze(out)


def diff(data: _np.ndarray, frame1, frame2=None, mode: COMBINE_MODE = COMBINE_MODE.FULL, precision='f64'):
    """High order combination with combine function difference. e.g. x-y.

    Args:
        data (ndarray): 1-D/2-D input samples.
        frame1 (slice or list): first traces frame. 
        frame2 (slice or list): second traces frame. Defaults to None, reuse frame1.
        mode (str, optional): Defaults to 'full'.  
            In SAME mode, every sample of `frame1` will be combined with the corresponding sample in frame2. 
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x1·y1, x2·y2]
            In FULL mode, every sample of `frame1` is combined with full `frame2`.
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x0·y1, x0·y2, x1·y0, x1·y1, x1·y2, x2·y0, x2·y1, x2·y2]
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
    """
    dtype = get_precision(precision)
    frame1, frame2 = _check_frames(frame1, frame2, mode)
    nb_samples = data.shape[-1]
    data = data.reshape(-1, nb_samples)
    fs1 = data[:, frame1].astype(dtype)
    fs2 = data[:, frame2].astype(dtype)
    if mode == COMBINE_MODE.SAME:
        out = fs1 - fs2
    else:
        if precision == 'f64':
            out = _n_rust.combine_diff_r(nuscar._global_pool, fs1, fs2)
        else:
            out = _n_rust.combine_diff_r32(nuscar._global_pool, fs1, fs2)
    return _np.squeeze(out)


def abs_diff(data: _np.ndarray, frame1, frame2=None, mode: COMBINE_MODE = COMBINE_MODE.FULL, precision='f64'):
    """High order combination with combine function absolute difference. e.g. |x-y|.

    Args:
        data (ndarray): 1-D/2-D input samples.
        frame1 (slice or list): first traces frame. 
        frame2 (slice or list): second traces frame. Defaults to None, reuse frame1.
        mode (str, optional): Defaults to 'full'.  
            In SAME mode, every sample of `frame1` will be combined with the corresponding sample in frame2. 
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x1·y1, x2·y2]
            In FULL mode, every sample of `frame1` is combined with full `frame2`.
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x0·y1, x0·y2, x1·y0, x1·y1, x1·y2, x2·y0, x2·y1, x2·y2]
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
    """
    dtype = get_precision(precision)
    frame1, frame2 = _check_frames(frame1, frame2, mode)
    nb_samples = data.shape[-1]
    data = data.reshape(-1, nb_samples)
    fs1 = data[:, frame1].astype(dtype)
    fs2 = data[:, frame2].astype(dtype)
    if mode == COMBINE_MODE.SAME:
        out = _np.abs(fs1 - fs2)
    else:
        if precision == 'f64':
            out = _n_rust.combine_abs_diff_r(nuscar._global_pool, fs1, fs2)
        else:
            out = _n_rust.combine_abs_diff_r32(nuscar._global_pool, fs1, fs2)
    return _np.squeeze(out)


def center_product(data: _np.ndarray, frame1, frame2=None, mode: COMBINE_MODE = COMBINE_MODE.FULL, mean=None, precision='f64'):
    """High order combination with combine function center product. e.g. (x-E(x))*(y-E(y)).

    Args:
        data (ndarray): 1-D/2-D input samples.
        frame1 (slice or list): first traces frame. 
        frame2 (slice or list): second traces frame. Defaults to None, reuse frame1.
        mode (str, optional): Defaults to 'full'.  
            In SAME mode, every sample of `frame1` will be combined with the corresponding sample in frame2. 
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x1·y1, x2·y2]
            In FULL mode, every sample of `frame1` is combined with full `frame2`.
                e.g. frame1 = [x0, x1, x2], frame2 = [y0, y1, y2], output will be [x0·y0, x0·y1, x0·y2, x1·y0, x1·y1, x1·y2, x2·y0, x2·y1, x2·y2]
        mean(ndarray): mean of traces. If None, the mean of batch traces will be used.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
    """
    dtype = get_precision(precision)
    frame1, frame2 = _check_frames(frame1, frame2, mode)
    nb_samples = data.shape[-1]
    data = data.reshape(-1, nb_samples)
    fs1 = data[:, frame1].astype(dtype)
    fs2 = data[:, frame2].astype(dtype)
    if mean is None:
        mean1 = _np.mean(fs1, axis=0)
        mean2 = _np.mean(fs2, axis=0)
    else:
        mean1 = mean[frame1].astype(dtype)
        mean2 = mean[frame2].astype(dtype)

    if mode == COMBINE_MODE.SAME:
        out = (fs1 - mean1) * (fs2 - mean2)
    else:
        if precision == 'f64':
            out = _n_rust.combine_center_product_r(nuscar._global_pool, fs1, fs2, mean1, mean2)
        else:
            out = _n_rust.combine_center_product_r32(nuscar._global_pool, fs1, fs2, mean1, mean2)
    return _np.squeeze(out)
