# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from tqdm.notebook import tqdm
import numpy as _np
import nuscar.nuscar_rust as _n_rust
from scipy.signal import butter, sosfilt, find_peaks, savgol_filter, iirnotch, filtfilt
import enum
from sklearn.decomposition import PCA as _PCA
import nuscar

class PATTERN_ALG(enum.IntEnum):
    """Enumeration pattern detect."""

    CORRELATION = 0
    DISTANCE = 1

def get_precision(precision: str):
    """Return the NumPy dtype for a precision string.

    Args:
        precision (str): ``'f64'`` for ``numpy.float64`` or ``'f32'`` for ``numpy.float32``.

    Returns:
        numpy.dtype: Floating-point dtype matching ``precision``.
    """
    if precision == 'f64':
        return _np.float64
    elif precision == 'f32':
        return _np.float32
    else:
        raise ValueError("precision must be 'f64' or 'f32'")

def filter_lowpass(data: _np.ndarray, fs: float, cutoff: float, order: int = 3, precision='f64',
                   batch_size: int = 0, show_progress: bool = False) -> _np.ndarray:
    """lowpass filter

    Args:
        data (numpy.ndarray): input samples
        fs (float): sample rate frequency
        cutoff (float): cutoff frequency
        order (int, optional): Defaults to 3.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
        batch_size (int, optional): traces per batch. 0 = all at once. Defaults to 0.
        show_progress (bool, optional): show progress bar. Defaults to False.

    Returns:
        numpy.ndarray
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    nb_traces = data.shape[0]

    if show_progress and batch_size <= 0:
        batch_size = max(1, nb_traces // 100)

    if batch_size > 0 and nb_traces > batch_size:
        output = _np.empty_like(data)
        pbar = tqdm(total=nb_traces, disable=not show_progress)
        for start in range(0, nb_traces, batch_size):
            end = min(start + batch_size, nb_traces)
            batch = data[start:end]
            if precision == 'f64':
                output[start:end] = _n_rust.low_pass_r(nuscar._global_pool, order, batch, fs, cutoff)
            else:
                sos = butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
                output[start:end] = _np.array([sosfilt(sos, row) for row in batch], dtype=dtype)
            pbar.update(end - start)
        pbar.close()
        return _np.squeeze(output)

    if precision == 'f64':
        filtered = _n_rust.low_pass_r(nuscar._global_pool, order, data, fs, cutoff)
    else:
        sos = butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
        filtered = _np.array([sosfilt(sos, row) for row in data], dtype=dtype)
    return _np.squeeze(filtered)


def filter_highpass(data: _np.ndarray, fs: float, cutoff: float, order: int = 3, precision='f64',
                    batch_size: int = 0, show_progress: bool = False) -> _np.ndarray:
    """highpass filter

    Args:
        data (numpy.ndarray): input samples
        fs (float): sample rate frequency
        cutoff (float): cutoff frequency
        order (int, optional): Defaults to 3.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
        batch_size (int, optional): traces per batch. 0 = all at once. Defaults to 0.
        show_progress (bool, optional): show progress bar. Defaults to False.

    Returns:
        numpy.ndarray
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    nb_traces = data.shape[0]

    if show_progress and batch_size <= 0:
        batch_size = max(1, nb_traces // 100)

    if batch_size > 0 and nb_traces > batch_size:
        output = _np.empty_like(data)
        pbar = tqdm(total=nb_traces, disable=not show_progress)
        for start in range(0, nb_traces, batch_size):
            end = min(start + batch_size, nb_traces)
            batch = data[start:end]
            if precision == 'f64':
                output[start:end] = _n_rust.high_pass_r(nuscar._global_pool, order, batch, fs, cutoff)
            else:
                sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')
                output[start:end] = _np.array([sosfilt(sos, row) for row in batch], dtype=dtype)
            pbar.update(end - start)
        pbar.close()
        return _np.squeeze(output)

    if precision == 'f64':
        filtered = _n_rust.high_pass_r(nuscar._global_pool, order, data, fs, cutoff)
    else:
        sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')
        filtered = _np.array([sosfilt(sos, row) for row in data], dtype=dtype)
    return _np.squeeze(filtered)


def filter_bandpass(data: _np.ndarray, fs: float, cutoff: tuple, order: int = 3, precision='f64',
                    batch_size: int = 0, show_progress: bool = False) -> _np.ndarray:
    """bandpass filter

    Args:
        data (numpy.ndarray): input samples
        fs (float): sample rate frequency
        cutoff (lowcut, highcut): cutoff frequency
        order (int, optional): Defaults to 3.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
        batch_size (int, optional): traces per batch. 0 = all at once. Defaults to 0.
        show_progress (bool, optional): show progress bar. Defaults to False.

    Returns:
        numpy.ndarray
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    nb_traces = data.shape[0]

    if show_progress and batch_size <= 0:
        batch_size = max(1, nb_traces // 100)

    if batch_size > 0 and nb_traces > batch_size:
        output = _np.empty_like(data)
        pbar = tqdm(total=nb_traces, disable=not show_progress)
        for start in range(0, nb_traces, batch_size):
            end = min(start + batch_size, nb_traces)
            batch = data[start:end]
            if precision == 'f64':
                output[start:end] = _n_rust.band_pass_r(nuscar._global_pool, order, batch, fs, cutoff[0], cutoff[1])
            else:
                sos = butter(order, [cutoff[0], cutoff[1]], btype='bandpass', fs=fs, output='sos')
                output[start:end] = _np.array([sosfilt(sos, row) for row in batch], dtype=dtype)
            pbar.update(end - start)
        pbar.close()
        return _np.squeeze(output)

    if precision == 'f64':
        filtered = _n_rust.band_pass_r(nuscar._global_pool, order, data, fs, cutoff[0], cutoff[1])
    else:
        sos = butter(order, [cutoff[0], cutoff[1]], btype='bandpass', fs=fs, output='sos')
        filtered = _np.array([sosfilt(sos, row) for row in data], dtype=dtype)
    return _np.squeeze(filtered)


def filter_bandstop(data: _np.ndarray, fs: float, cutoff: tuple, order: int = 3, precision='f64',
                    batch_size: int = 0, show_progress: bool = False) -> _np.ndarray:
    """bandstop filter

    Args:
        data (numpy.ndarray): input samples
        fs (float): sample rate frequency
        cutoff (lowcut, highcut): cutoff frequency
        order (int, optional): Defaults to 3.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
        batch_size (int, optional): traces per batch. 0 = all at once. Defaults to 0.
        show_progress (bool, optional): show progress bar. Defaults to False.

    Returns:
        numpy.ndarray
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    nb_traces = data.shape[0]

    if show_progress and batch_size <= 0:
        batch_size = max(1, nb_traces // 100)

    if batch_size > 0 and nb_traces > batch_size:
        output = _np.empty_like(data)
        pbar = tqdm(total=nb_traces, disable=not show_progress)
        for start in range(0, nb_traces, batch_size):
            end = min(start + batch_size, nb_traces)
            batch = data[start:end]
            if precision == 'f64':
                output[start:end] = _n_rust.band_stop_r(nuscar._global_pool, order, batch, fs, cutoff[0], cutoff[1])
            else:
                sos = butter(order, [cutoff[0], cutoff[1]], btype='bandstop', fs=fs, output='sos')
                output[start:end] = _np.array([sosfilt(sos, row) for row in batch], dtype=dtype)
            pbar.update(end - start)
        pbar.close()
        return _np.squeeze(output)

    if precision == 'f64':
        filtered = _n_rust.band_stop_r(nuscar._global_pool, order, data, fs, cutoff[0], cutoff[1])
    else:
        sos = butter(order, [cutoff[0], cutoff[1]], btype='bandstop', fs=fs, output='sos')
        filtered = _np.array([sosfilt(sos, row) for row in data], dtype=dtype)
    return _np.squeeze(filtered)


def filter_smooth(data: _np.ndarray, window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0, axis: int = -1, mode: str = 'interp', cval: float = 0.0, precision='f64') -> _np.ndarray:
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
      1. Light smoothing (preserves more details)
      - window_length: 5-7
      - polyorder: 2-3
      2. Medium smoothing (balances noise and signal)
      - window_length: 9-15
      - polyorder: 3-4
      3. Heavy smoothing (strong denoising)
      - window_length: 17-31
      - polyorder: 4-5

      Important rules:
      - window_length must be odd
      - polyorder must be < window_length
      - typically polyorder is set between 2-5
    Args:
        data (numpy.ndarray): input samples
        window_length (int): The length of the filter window (i.e., the number of coefficients).
            Must be a positive odd integer.
        polyorder (int): The order of the polynomial used to fit the samples.
            Must be less than window_length.
        deriv (int, optional): The order of the derivative to compute. Default is 0, which
            means to filter the data without differentiating.
        delta (float, optional): The spacing of the samples to which the filter will be applied.
            Used only if deriv > 0. Default is 1.0.
        axis (int, optional): The axis of the array along which to apply the filter.
            Default is -1.
        mode (str, optional): Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
            Default is 'interp'.
        cval (float, optional): Value to fill past the edges of the input if mode is 'constant'.
            Default is 0.0.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.

    Returns:
        numpy.ndarray: The filtered data.
    """
    dtype = get_precision(precision)
    data = data.astype(dtype, copy=False)
    
    filtered = savgol_filter(data, window_length, polyorder, deriv=deriv, delta=delta, 
                            axis=axis, mode=mode, cval=cval)
    
    return filtered.astype(dtype, copy=False)


def filter_harmonic(data: _np.ndarray, fs: float, fundamental_freq: float, harmonics: int = 5, 
                    q_factor: float = 30.0, mode: str = 'remove', precision='f64') -> _np.ndarray:
    """Harmonic filtering to remove or extract fundamental frequency and its harmonics.
    
    Args:
        data (numpy.ndarray): input samples
        fs (float): sample rate frequency in Hz
        fundamental_freq (float): fundamental frequency to filter in Hz
        harmonics (int, optional): number of harmonics to filter (including fundamental). Defaults to 5.
        q_factor (float, optional): quality factor for notch filter (higher = narrower notch). Defaults to 30.0.
        mode (str, optional): 'remove' to remove harmonics, 'extract' to keep only harmonics. Defaults to 'remove'.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.
    
    Returns:
        numpy.ndarray: filtered data
    
    Example:
        # Remove 50Hz power line noise and its harmonics (100Hz, 150Hz, 200Hz, 250Hz)
        filtered = filter_harmonic(data, fs=1000, fundamental_freq=50, harmonics=5, mode='remove')
        
        # Extract only 60Hz and its harmonics 
        filtered = filter_harmonic(data, fs=1000, fundamental_freq=60, harmonics=3, mode='extract')
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    
    filtered_data = data.copy()
    
    for trace_idx in range(data.shape[0]):
        trace = data[trace_idx]
        
        if mode == 'remove':
            # Remove harmonics using cascaded notch filters
            for harmonic_n in range(1, harmonics + 1):
                freq = fundamental_freq * harmonic_n
                if freq < fs / 2:  # Only filter frequencies below Nyquist
                    b, a = iirnotch(freq, q_factor, fs)
                    trace = filtfilt(b, a, trace)
            filtered_data[trace_idx] = trace
            
        elif mode == 'extract':
            # Extract harmonics using bandpass filters around each harmonic
            extracted = _np.zeros_like(trace)
            for harmonic_n in range(1, harmonics + 1):
                freq = fundamental_freq * harmonic_n
                if freq < fs / 2:  # Only process frequencies below Nyquist
                    # Create narrow bandpass filter around harmonic
                    bandwidth = freq / q_factor
                    lowcut = max(freq - bandwidth/2, 0.1)
                    highcut = min(freq + bandwidth/2, fs/2 - 0.1)
                    
                    if highcut > lowcut:
                        sos = butter(4, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
                        harmonic_component = sosfilt(sos, trace)
                        extracted += harmonic_component
            filtered_data[trace_idx] = extracted
        else:
            raise ValueError("mode must be 'remove' or 'extract'")
    
    return _np.squeeze(filtered_data)


def moving_mean(data: _np.ndarray, window_size: int, precision='f64') -> _np.ndarray:
    """ moving mean

    Args:
        data (numpy.ndarray): input samples
        window_size (int): window size of moving operation.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.

    Returns:
        numpy.ndarray 
    """
    nb_samples = data.shape[-1]
    data = data.astype(get_precision(precision), copy=False).reshape(-1, nb_samples)
    if precision == 'f64':
        return _np.squeeze(_n_rust.moving_mean_r(nuscar._global_pool, data, window_size))
    else:
        return _np.squeeze(_n_rust.moving_mean_r32(nuscar._global_pool, data, window_size))


def moving_var(data: _np.ndarray, window_size: int, precision='f64') -> _np.ndarray:
    """ moving variance

    Args:
        data (numpy.ndarray): input samples
        window_size (int): window size of moving operation.
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.

    Returns:
        numpy.ndarray 
    """
    nb_samples = data.shape[-1]
    data = data.astype(get_precision(precision), copy=False).reshape(-1, nb_samples)
    if precision == 'f64':
        return _np.squeeze(_n_rust.moving_var_r(nuscar._global_pool, data, window_size))
    else:
        return _np.squeeze(_n_rust.moving_var_r32(nuscar._global_pool, data, window_size))


def moving_skew(data, window_size, precision='f64'):
    """moving skew

    Args:
        data (numpy.ndarray): input samples
        window_size (int): window size of moving operation
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.

    Returns:
        numpy.ndarray

    """
    data = _np.asarray(data, dtype=get_precision(precision))
    m1 = moving_mean(data, window_size, precision)
    m2 = moving_mean(data**2, window_size, precision)
    m3 = moving_mean(data**3, window_size, precision)
    v = m2 - m1**2
    return _np.squeeze((m3 - 3 * m1 * v - m1**3) / v**(3 / 2))


def moving_kurtosis(data, window_size, precision='f64'):
    """moving kurtosis

    Args:
        data (numpy.ndarray): input samples
        window_size (int): window size of moving operation
        precision (str, optional): Defaults to 'f64'. 'f64' for 64-bit float, 'f32' for 32-bit float.

    Returns:
        numpy.ndarray

    """
    data = _np.asarray(data, dtype=get_precision(precision))
    m1 = moving_mean(data, window_size, precision)
    m2 = moving_mean(data**2, window_size, precision)
    m3 = moving_mean(data**3, window_size, precision)
    m4 = moving_mean(data**4, window_size, precision)
    v = m2 - m1**2

    return _np.squeeze((m4 - 4 * m3 * m1 + 6 * v * m1**2 + 3 * m1**4) / v**2 - 3)


def fft(data: _np.ndarray, frequency: float):
    """FFT on 1-D/2-D ndarray

    Args:
        data (numpy.ndarray): input samples.
        frequency (float): the sampling rate/frequency, in Hz.

    Returns:
        (numpy.ndarray, numpy.ndarray): frequencies, magnitude.
    """

    nb_samples = data.shape[-1]
    frequencies = _np.fft.rfftfreq(nb_samples) * frequency
    fft = _np.fft.rfft(data, axis=-1) / nb_samples
    magnitude = _np.abs(fft)
    return frequencies, magnitude


def best_interval(data: _np.ndarray, win_len: int, use_abs: bool = True) -> tuple:
    """Find the contiguous window of length win_len that maximizes the sum of values.

    Useful for selecting the best POI (Point of Interest) interval from
    correlation traces. Supports both single-trace and multi-trace input.

    When multiple traces are provided (2-D or higher), the function first
    aggregates them into a single score curve by taking the absolute value
    of each trace and summing across all non-sample axes. This ensures the
    selected interval captures leakage from ALL target bytes, not just one.

    Args:
        data (numpy.ndarray): input signal.
            - 1-D ``(n_samples,)``: single correlation trace.
            - 2-D ``(n_traces, n_samples)``: multiple traces, e.g. one per
              target byte from a multi-target CPA.
            - Higher dimensions are reshaped to ``(-1, n_samples)`` automatically,
              so passing ``task.result`` with shape ``(n_guesses, n_targets, n_samples)``
              directly is supported.
        win_len (int): window length in samples.
        use_abs (bool, optional): take absolute values before scoring. Defaults to True.

    Returns:
        (int, int): (start, end) indices of the best interval.

    Example:
        >>> # Single trace
        >>> start, end = best_interval(task.result, win_len=100)
        >>>
        >>> # Multi-target: task.result shape = (n_targets, n_samples)
        >>> start, end = best_interval(task.result, win_len=40)
    """
    x = _np.asarray(data, dtype=_np.float64)

    # Multi-trace: aggregate into a single score curve
    if x.ndim >= 2:
        n_samples = x.shape[-1]
        x = x.reshape(-1, n_samples)   # (n_curves, n_samples)
        x = _np.abs(x).sum(axis=0)     # (n_samples,)
    else:
        x = x.reshape(-1)

    if use_abs:
        x = _np.abs(x)

    if win_len <= 0 or win_len > x.size:
        raise ValueError(f"win_len must be in [1, {x.size}]")

    kernel = _np.ones(int(win_len), dtype=_np.float64)
    score = _np.convolve(x, kernel, mode="valid")
    start = int(_np.argmax(score))
    end = start + int(win_len)
    return start, end


def peak_detect(data: _np.ndarray, height: float = None, distance: int = None, precision='f64'):
    """find peak of 1-D ndarray.

    Args:
        data (_np.ndarray): input samples.
        height (float): min height of a peak.
        distance (int): minimal horizontal distance in samples between neighbouring peaks.
    """
    if not data.ndim == 1:
        raise ValueError("data must be a 1D ndarray.")
    dtype = get_precision(precision)
    peak, _ = find_peaks(data.astype(dtype, copy=False), height=height, distance=distance)
    return peak


def pattern_detect(data: _np.ndarray, pattern: _np.ndarray, alg: PATTERN_ALG = PATTERN_ALG.CORRELATION, precision='f64'):
    """find pattern of 1-D/2-D ndarray.

    Args:
        data (ndarray): 1-D/2-D input samples.
        pattern (ndarray): 1-D pattern.
        alg (PATTERN_ALG): CORRELATION or DISTANCE
    """
    dtype = get_precision(precision)
    nb_samples = data.shape[-1]
    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    pattern = pattern.astype(dtype, copy=False)
    if not pattern.ndim == 1:
        raise ValueError("pattern must be a 1D ndarray.")
    if len(pattern) > nb_samples:
        raise ValueError("pattern length must be less than trace length.")

    data = data.astype(dtype, copy=False).reshape(-1, nb_samples)
    if alg == PATTERN_ALG.CORRELATION:
        if precision == 'f64':
            return _np.squeeze(_n_rust.pattern_corr_r(nuscar._global_pool, data, pattern))
        else:
            return _np.squeeze(_n_rust.pattern_corr_r32(nuscar._global_pool, data, pattern))
    elif alg == PATTERN_ALG.DISTANCE:
        if precision == 'f64':
            return _np.squeeze(_n_rust.pattern_dist_r(nuscar._global_pool, data, pattern))
        else:
            return _np.squeeze(_n_rust.pattern_dist_r32(nuscar._global_pool, data, pattern))

def extract_around_peak(data: _np.ndarray, peak: _np.ndarray, before: int, after: int):
    """Extract flattened windows around peak positions from a 1-D signal.

    Args:
        data (ndarray): 1-D input signal.
        peak (ndarray): 1-D array of peak indices.
        before (int): Number of samples to include before each peak.
        after (int): Number of samples to include after each peak.

    Returns:
        ndarray: Flattened samples from all peak-centered windows.
    """
    if not peak.ndim == 1:
        raise ValueError("peak must be a 1D ndarray.")
    if not data.ndim == 1:
        raise ValueError("data must be a 1D ndarray.")
    s = _np.empty((len(peak), before+after), dtype=data.dtype)
    extended_indexes = _np.tile(
        _np.arange(-before, after + 1), (len(peak), 1))
    extended_indexes = (extended_indexes.T + peak).T
    extended_indexes = _np.mod(extended_indexes, len(data))
    return _np.take(data, extended_indexes).reshape((-1,))


def pca_decomposition(data: _np.ndarray, n_components: int):
    """Principal component analysis to reduce dimension.

    Args:
        data (ndarray): input samples, with shape (nb_traces, nb_sampes).
        n_components (int): components to keep.
    """
    pca = _PCA(n_components=n_components)
    return pca.fit_transform(data)


def elastic_align(ref: _np.ndarray, data: _np.ndarray, radius: int = 1,
                  batch_size: int = 0, show_progress: bool = True):
    """Elastic Alignment to align traces using FastDTW (Rust implementation).

    Uses Rayon-parallelized batch processing for multi-trace alignment.

    Args:
        ref (ndarray): 1-D trace, for reference.
        data (_np.ndarray): 1-D/2-D trace set, will align to ref.
        radius (int): Search radius for FastDTW. Default is 1.
        batch_size (int): number of traces per batch. 0 = all at once.
        show_progress (bool): show tqdm progress bar (auto-enables batching if batch_size=0).
    """

    nb_samples = data.shape[-1]
    data = data.reshape(-1, nb_samples)

    # Ensure ref and data are float64 for Rust function
    ref = _np.ascontiguousarray(ref, dtype=_np.float64)
    data = _np.ascontiguousarray(data, dtype=_np.float64)

    nb_traces = data.shape[0]

    if show_progress and batch_size <= 0:
        batch_size = max(nb_traces // 100, min(nb_traces, 100))

    if batch_size > 0 and nb_traces > batch_size:
        output = _np.empty_like(data)
        pbar = tqdm(total=nb_traces, disable=not show_progress)
        for start in range(0, nb_traces, batch_size):
            end = min(start + batch_size, nb_traces)
            output[start:end] = _n_rust.elastic_align_batch_r(ref, data[start:end], radius)
            pbar.update(end - start)
        pbar.close()
        return _np.squeeze(output)

    out = _n_rust.elastic_align_batch_r(ref, data, radius)
    return _np.squeeze(out)
