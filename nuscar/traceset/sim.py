# Copyright(c)  2026. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

"""Synthetic (simulated) trace generation.

These helpers build a :class:`~nuscar_pro.traceset.ContainerMemory` filled with
artificial leakage traces, so attacks/metrics/tutorials and unit tests can run
without real acquisition hardware. Traces are produced by embedding a leakage
value (e.g. Hamming weight of an intermediate) at chosen sample positions and
adding Gaussian noise. Because the result is a normal ``ContainerMemory`` it
plugs directly into every existing task, viewer and distinguisher.

For datasets too large to hold in memory, ``simulate_traces_to_disk`` and
``simulate_aes_traces_to_disk`` stream batches straight to a directory of
``.npy`` files (loadable via :class:`~nuscar_pro.traceset.ContainerNPY`).
Each trace's randomness is derived from an independent RNG seeded by
``(seed, trace_index)``, so any single trace is reproducible on its own,
regardless of batch size, generation order, or parallel worker count.
"""

import logging
import shutil as _shutil
from pathlib import Path as _Path

import joblib as _joblib
import numpy as _np
from numpy.lib.format import open_memmap as _open_memmap
from tqdm.notebook import tqdm as _tqdm
import nuscar
from nuscar.leakmodel import leakage_model_hw as _leakage_model_hw
from nuscar.traceset.mem import StorerMemory, ContainerMemory

logger = logging.getLogger(__name__.split('.')[0])


def _default_poi(nb_samples: int, nb_points: int) -> _np.ndarray:
    """Compute evenly-spread, guaranteed-distinct default leak positions.

    Spreads ``nb_points`` integer positions across ``[0.1, 0.9] * nb_samples``
    and nudges any position that would collide (after rounding) forward to the
    next free slot, so the result is always strictly increasing.

    Raises:
        ValueError: if ``nb_samples`` is too small to fit ``nb_points`` distinct
            positions within that margin.
    """
    lo, hi = nb_samples * 0.1, nb_samples * 0.9
    poi = _np.round(_np.linspace(lo, hi, nb_points)).astype(int)
    for i in range(1, nb_points):
        if poi[i] <= poi[i - 1]:
            poi[i] = poi[i - 1] + 1
    if poi[0] < 0 or poi[-1] >= nb_samples or len(set(poi.tolist())) != nb_points:
        raise ValueError(
            f"cannot place {nb_points} distinct default poi positions within "
            f"nb_samples={nb_samples} (evenly spread across the [0.1, 0.9] "
            f"margin); increase nb_samples or pass poi explicitly"
        )
    return poi


def _prepare_output_dir(output_dir, *, overwrite: bool) -> _Path:
    """Create ``output_dir``, refusing to clobber an existing one silently."""
    output_dir = _Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; remove it first or pass overwrite=True"
            )
        _shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def simulate_traces(leakages: _np.ndarray, nb_samples: int, poi=None,
                    noise: float = 1.0, seed=None, metadatas: dict = None):
    """Build a container of synthetic traces from a leakage matrix.

    Each column of ``leakages`` is embedded (added) at one sample position of an
    otherwise Gaussian-noise trace.

    Args:
        leakages (ndarray): shape ``(nb_traces, nb_points)`` float leakage values
            to embed into the traces (one column per point of interest).
        nb_samples (int): total number of samples per trace.
        poi (sequence of int, optional): sample positions where each leakage
            column is placed. Length must equal ``nb_points``. If ``None`` the
            points are spread evenly across ``[0.1, 0.9] * nb_samples``.
        noise (float): standard deviation of the additive Gaussian noise.
        seed (int, optional): seed for the random generator (noise + defaults).
        metadatas (dict, optional): extra metadata arrays keyed by name, each of
            shape ``(nb_traces, ...)``. Stored alongside the traces.

    Returns:
        ContainerMemory: container exposing ``samples`` and the given metadata.
            The chosen ``poi`` positions are attached as the ``.poi`` attribute.
    """
    rng = _np.random.default_rng(seed)

    leakages = _np.asarray(leakages, dtype=_np.float64)
    if leakages.ndim != 2:
        raise ValueError(
            f"leakages must be 2-D (nb_traces, nb_points), got shape {leakages.shape}")
    nb_traces, nb_points = leakages.shape

    if nb_samples < nb_points:
        raise ValueError(
            f"nb_samples ({nb_samples}) must be >= number of leakage points ({nb_points})")

    if poi is None:
        poi = _default_poi(nb_samples, nb_points)
    else:
        poi = _np.asarray(poi, dtype=int).reshape(-1)
        if poi.shape[0] != nb_points:
            raise ValueError(
                f"poi must have {nb_points} positions (one per leakage column), got {poi.shape[0]}")
        if poi.min() < 0 or poi.max() >= nb_samples:
            raise ValueError(
                f"poi positions must be in [0, {nb_samples - 1}], "
                f"got min={poi.min()}, max={poi.max()}")
        if len(set(poi.tolist())) != nb_points:
            raise ValueError(
                "poi positions must be distinct; overlapping positions would sum "
                "more than one leakage column into the same sample"
            )

    traces = rng.normal(0.0, noise, size=(nb_traces, nb_samples))
    for j in range(nb_points):
        traces[:, poi[j]] += leakages[:, j]

    store = StorerMemory()
    meta = {}
    if metadatas is not None:
        for name, arr in metadatas.items():
            arr = _np.asarray(arr)
            if arr.shape[0] != nb_traces:
                raise ValueError(
                    f"metadata '{name}' has {arr.shape[0]} rows, expected {nb_traces}")
            if arr.ndim == 1:
                arr = arr[:, None]
            meta[name] = arr
    store.update(samples=traces, **meta)

    container = ContainerMemory(store)
    container.poi = poi
    return container


def simulate_traces_to_disk(output_dir, nb_traces: int, nb_samples: int, gen_leak,
                            poi=None, nb_points: int = None, noise: float = 1.0,
                            seed: int = 0, batch_size: int = None, n_jobs: int = 1,
                            dtype=_np.float32, overwrite: bool = False):
    """Stream synthetic traces straight to a directory of ``.npy`` files.

    Unlike :func:`simulate_traces` (which builds one in-memory leakage matrix
    up front), this writes batch by batch via memory-mapped ``.npy`` files, so
    datasets far larger than RAM (e.g. millions of traces) can be generated.

    Each trace's additive noise is drawn from an independent
    ``np.random.default_rng((seed, trace_index))``, so regenerating any single
    trace index always reproduces the same noise regardless of ``batch_size``,
    generation order, or ``n_jobs``. ``gen_leak`` is expected to derive its own
    per-trace randomness (if any) the same way -- see
    :func:`simulate_aes_traces_to_disk` for a worked example.

    Args:
        output_dir (str or Path): destination directory; created if missing.
            Existing contents are only replaced when ``overwrite=True``.
        nb_traces (int): total number of traces to generate.
        nb_samples (int): samples per trace.
        gen_leak (callable): ``gen_leak(indices, seed) -> (leakages, metadatas)``
            called once per batch, where ``indices`` is a 1-D int array of trace
            indices. Must return ``leakages`` of shape
            ``(len(indices), nb_points)`` and a ``metadatas`` dict of arrays each
            shaped ``(len(indices), ...)``. The same set of metadata keys must be
            returned on every call.
        poi (sequence of int, optional): sample positions for each leak point.
            Evenly spread across ``[0.1, 0.9] * nb_samples`` if ``None``.
        nb_points (int, optional): number of leak points; required when ``poi``
            is ``None`` (otherwise inferred from ``len(poi)``).
        noise (float): standard deviation of the additive Gaussian noise.
        seed (int): base seed; combined with each trace index to derive that
            trace's independent RNG.
        batch_size (int, optional): traces written per batch. Auto-sized from
            ``nb_samples`` (via ``nuscar_pro._find_batch_size``) if ``None``.
        n_jobs (int, optional): parallel workers for the per-trace noise draws
            within a batch. Defaults to 1 (serial). Use -1 for all cores.
            Uses joblib's threading backend, consistent with
            ``Storer.update_sync``.
        dtype (numpy dtype): on-disk sample dtype. Defaults to float32 to bound
            disk usage for large datasets.
        overwrite (bool): remove an existing ``output_dir`` before writing.

    Returns:
        Path: ``output_dir``, containing ``samples.npy``, ``poi.npy`` and one
            ``.npy`` file per metadata field. Load with
            ``ContainerNPY.from_dir(output_dir, mmap_mode='r')``.
    """
    if nb_traces <= 0:
        raise ValueError("nb_traces must be positive")
    if poi is None and nb_points is None:
        raise ValueError("either poi or nb_points must be given")

    if poi is None:
        poi = _default_poi(nb_samples, nb_points)
    else:
        poi = _np.asarray(poi, dtype=int).reshape(-1)
        if nb_points is not None and poi.shape[0] != nb_points:
            raise ValueError(
                f"poi has {poi.shape[0]} positions but nb_points={nb_points}")
        nb_points = poi.shape[0]
        if poi.min() < 0 or poi.max() >= nb_samples:
            raise ValueError(
                f"poi positions must be in [0, {nb_samples - 1}], "
                f"got min={poi.min()}, max={poi.max()}")
        if len(set(poi.tolist())) != nb_points:
            raise ValueError(
                "poi positions must be distinct; overlapping positions would sum "
                "more than one leakage column into the same sample"
            )

    output_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    _np.save(output_dir / "poi.npy", poi)

    if batch_size is None:
        batch_size = nuscar._find_batch_size(nb_samples)

    samples_mm = _open_memmap(
        output_dir / "samples.npy", mode="w+", dtype=dtype,
        shape=(nb_traces, nb_samples))

    def _noise_row(idx):
        rng = _np.random.default_rng((seed, int(idx)))
        return rng.normal(0.0, noise, size=nb_samples)

    meta_mm = {}
    meta_keys = None
    pbar = _tqdm(total=nb_traces)
    for start in range(0, nb_traces, batch_size):
        stop = min(start + batch_size, nb_traces)
        indices = _np.arange(start, stop)
        count = stop - start

        leakages, metadatas = gen_leak(indices, seed)
        leakages = _np.asarray(leakages, dtype=_np.float64)
        if leakages.shape != (count, nb_points):
            raise ValueError(
                f"gen_leak must return leakages of shape {(count, nb_points)}, "
                f"got {leakages.shape}")

        if meta_keys is None:
            meta_keys = set(metadatas.keys())
            for name, arr in metadatas.items():
                arr = _np.asarray(arr)
                meta_mm[name] = _open_memmap(
                    output_dir / f"{name}.npy", mode="w+", dtype=arr.dtype,
                    shape=(nb_traces,) + arr.shape[1:])
        elif set(metadatas.keys()) != meta_keys:
            raise KeyError("gen_leak returned inconsistent metadata keys across batches")

        if n_jobs == 1:
            noise_batch = _np.stack([_noise_row(i) for i in indices])
        else:
            noise_batch = _np.stack(_joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                _joblib.delayed(_noise_row)(i) for i in indices))

        for j in range(nb_points):
            noise_batch[:, poi[j]] += leakages[:, j]

        samples_mm[start:stop] = noise_batch.astype(dtype, copy=False)
        for name, mm in meta_mm.items():
            mm[start:stop] = metadatas[name]

        pbar.update(count)
    pbar.close()

    samples_mm.flush()
    for mm in meta_mm.values():
        mm.flush()
    logger.info(f"Wrote {nb_traces} traces to {output_dir}")
    return output_dir


def simulate_aes_traces(nb_traces: int = 2000, nb_samples: int = 200, key=None,
                        target: str = 'sbox', noise: float = 1.0, poi=None,
                        seed=None):
    """Simulate AES-128 power traces leaking the Hamming weight of a first-round
    intermediate (one leak point per key byte).

    The generated container carries ``plaintext`` and ``key`` metadata, so it can
    be attacked directly with the selection functions in
    :mod:`nuscar_pro.ciphers.aes` (e.g. ``attack_first_sbox_hw``) to recover the
    key that was used to generate it.

    Args:
        nb_traces (int): number of traces to generate.
        nb_samples (int): samples per trace.
        key (array-like, optional): 16-byte AES key. Random if ``None``.
        target (str): intermediate to leak: ``'sbox'`` (first-round S-box output)
            or ``'addrk'`` (first AddRoundKey output).
        noise (float): standard deviation of the additive Gaussian noise.
        poi (sequence of int, optional): 16 sample positions for the 16 key-byte
            leak points. Evenly spread if ``None``.
        seed (int, optional): seed for reproducibility.

    Returns:
        ContainerMemory: container with ``plaintext``/``key`` metadata and a
            ``.poi`` attribute holding the 16 leak-point positions.
    """
    from nuscar.ciphers import aes as _aes

    rng = _np.random.default_rng(seed)

    if key is None:
        key = rng.integers(0, 256, size=16, dtype=_np.uint8)
    else:
        key = _np.asarray(key, dtype=_np.uint8).reshape(16)

    plaintext = rng.integers(0, 256, size=(nb_traces, 16), dtype=_np.uint8)
    key_tiled = _np.tile(key, (nb_traces, 1))

    if target == 'sbox':
        inter = _aes.compute_first_sbox_value(plaintext, key_tiled)
    elif target == 'addrk':
        inter = _aes.compute_first_addRk_value(plaintext, key_tiled)
    else:
        raise ValueError("target must be 'sbox' or 'addrk'")

    inter = _np.asarray(inter, dtype=_np.uint8).reshape(nb_traces, 16)
    leakages = _leakage_model_hw(inter).astype(_np.float64)  # (nb_traces, 16)

    return simulate_traces(
        leakages, nb_samples, poi=poi, noise=noise, seed=seed,
        metadatas={'plaintext': plaintext, 'key': key_tiled})


def simulate_aes_traces_to_disk(output_dir, nb_traces: int, nb_samples: int = 200,
                                key=None, target: str = 'sbox', noise: float = 1.0,
                                poi=None, seed: int = 0, batch_size: int = None,
                                n_jobs: int = 1, dtype=_np.float32,
                                overwrite: bool = False):
    """Stream a large AES-128 Hamming-weight-leakage dataset straight to disk.

    Same leakage model as :func:`simulate_aes_traces`, built on top of
    :func:`simulate_traces_to_disk` so datasets far larger than RAM (e.g.
    millions of traces) can be generated without materializing them in memory.

    Each trace's plaintext and noise are drawn from an independent RNG seeded
    by ``(seed, trace_index)``, so any single trace is reproducible on its own
    regardless of ``batch_size``, generation order, or ``n_jobs``; the
    deterministic AES/Hamming-weight computation is vectorized per batch for
    speed.

    Args:
        output_dir (str or Path): destination directory; created if missing.
            Existing contents are only replaced when ``overwrite=True``.
        nb_traces (int): number of traces to generate.
        nb_samples (int): samples per trace.
        key (array-like, optional): 16-byte AES key. Random (derived from
            ``seed``) if ``None``.
        target (str): intermediate to leak: ``'sbox'`` (first-round S-box
            output) or ``'addrk'`` (first AddRoundKey output).
        noise (float): standard deviation of the additive Gaussian noise.
        poi (sequence of int, optional): 16 sample positions for the 16
            key-byte leak points. Evenly spread if ``None``.
        seed (int): base seed; combined with each trace index to derive that
            trace's independent RNG, and (if ``key`` is ``None``) to derive the
            random key.
        batch_size (int, optional): traces written per batch. Auto-sized from
            ``nb_samples`` if ``None``.
        n_jobs (int, optional): parallel workers for the per-trace plaintext
            and noise draws within a batch. Defaults to 1 (serial). Use -1 for
            all cores. Uses joblib's threading backend, consistent with
            ``Storer.update_sync``.
        dtype (numpy dtype): on-disk sample dtype. Defaults to float32.
        overwrite (bool): remove an existing ``output_dir`` before writing.

    Returns:
        Path: ``output_dir``, containing ``samples.npy``, ``plaintext.npy``,
            ``key.npy``, ``poi.npy``. Load with
            ``ContainerNPY.from_dir(output_dir, mmap_mode='r')``.
    """
    from nuscar.ciphers import aes as _aes

    if target not in ('sbox', 'addrk'):
        raise ValueError("target must be 'sbox' or 'addrk'")

    if key is None:
        key = _np.random.default_rng(seed).integers(0, 256, size=16, dtype=_np.uint8)
    else:
        key = _np.asarray(key, dtype=_np.uint8).reshape(16)

    def _plaintext_row(idx):
        rng = _np.random.default_rng((seed, int(idx)))
        return rng.integers(0, 256, size=16, dtype=_np.uint8)

    def gen_leak(indices, seed):
        count = len(indices)
        if n_jobs == 1:
            plaintext = _np.stack([_plaintext_row(i) for i in indices])
        else:
            plaintext = _np.stack(_joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                _joblib.delayed(_plaintext_row)(i) for i in indices))

        key_tiled = _np.tile(key, (count, 1))
        if target == 'sbox':
            inter = _aes.compute_first_sbox_value(plaintext, key_tiled)
        else:
            inter = _aes.compute_first_addRk_value(plaintext, key_tiled)
        inter = _np.asarray(inter, dtype=_np.uint8).reshape(count, 16)
        leakages = _leakage_model_hw(inter).astype(_np.float64)
        return leakages, {'plaintext': plaintext}

    output_dir = simulate_traces_to_disk(
        output_dir, nb_traces=nb_traces, nb_samples=nb_samples, gen_leak=gen_leak,
        poi=poi, nb_points=16, noise=noise, seed=seed, batch_size=batch_size,
        n_jobs=n_jobs, dtype=dtype, overwrite=overwrite)

    _np.save(output_dir / "key.npy", key)
    return output_dir
