# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from pathlib import Path

import numpy as _np
from nuscar.traceset import Container
from nuscar.traceset import Storer


class StorerMemory(Storer):
    """In-memory trace storer for temporary or interactive collections."""

    def __init__(self):
        """Create an empty in-memory storer."""
        super().__init__()
        self._is_first_update = True
        self._mem_meta = dict()

    def _init_array(self, samples, **kwargs):
        self._samples = samples.reshape(-1, samples.shape[-1])
        for m in kwargs:
            d = kwargs[m]
            d = d.reshape(-1, d.shape[-1])
            self._mem_meta[m] =d
        self._is_first_update = False

    def update(self, samples: _np.array, **kwargs):
        """Append samples and metadata arrays to memory.

        Args:
            samples (ndarray): Trace samples to append.
            **kwargs: Metadata arrays keyed by metadata name.
        """
        super().update(samples, **kwargs)
        if self._is_first_update:
            self._init_array(samples, **kwargs)
        else:
            self._samples = _np.vstack((self._samples, samples))
            for m in kwargs:
                d = kwargs[m]
                self._mem_meta[m] = _np.vstack((self._mem_meta[m], d))
        
    def close(self):
        """Release resources held by the in-memory storer."""
        pass


class ContainerMemory(Container):
    """Container view over traces collected in a ``StorerMemory`` instance."""

    def __init__(self, memstore:StorerMemory, frame=None, func_preprocess=None):
        """Create a container for in-memory traces.

        Args:
            memstore (StorerMemory): Source in-memory storer.
            frame (slice, int, list, ndarray, range, optional): Sample frame to expose.
            func_preprocess (callable, optional): Preprocessing function applied to samples.
        """
        super().__init__(frame, func_preprocess)
        self._sub_traceset_indices = None
        self._memstore = memstore

    def __len__(self):
        if self._memstore._is_first_update:
            return 0
        if self._sub_traceset_indices is not None:
            return len(self._sub_traceset_indices)
        return len(self._memstore._samples)
    
    @ property
    def metadatas(self):
        return list(self._memstore._mem_meta)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = range(
                key.start if key.start is not None else 0,
                key.stop if key.stop is not None else len(self),
                key.step if key.step is not None else 1
            )
        sub_traceset_indices = self._convert_traces_indices_to_file_indices_array(
            traces=key)
        new_container = ContainerMemory(
            self._memstore, self._frame, self._func_pre)
        new_container._sub_traceset_indices = sub_traceset_indices
        if hasattr(self, 'poi'):
            # `.poi` (e.g. set by nuscar_pro.traceset.sim helpers) describes sample
            # positions, not per-trace data, so it is carried over as-is rather
            # than being indexed like per-trace metadata.
            new_container.poi = self.poi
        return new_container

    def __getattr__(self, name):
        try:
            assert (name in self.metadatas)
            if self._sub_traceset_indices is None:
                d = self._memstore._mem_meta[name][:]
            else:
                d = self._memstore._mem_meta[name][self._sub_traceset_indices]
            if len(d) == 1:
                return _np.squeeze(d)
            return d
        except:
            # or other errors that may occur
            raise AttributeError(name)

    def _convert_traces_indices_to_file_indices_array(self, traces):
        if self._sub_traceset_indices is not None:
            sub_max = len(self._sub_traceset_indices)
            traces_index = _np.array([t for t in traces if t < sub_max])
            return self._sub_traceset_indices[traces_index]
        else:
            sub_max = self.__len__()
            traces_index = _np.array([t for t in traces if t < sub_max])
            return _np.array(traces_index)

    @ property
    def _samples(self) -> _np.ndarray:
        if self._sub_traceset_indices is None:
            d = self._memstore._samples
        else:
            d = self._memstore._samples[self._sub_traceset_indices]
        return d

    @property
    def _nb_traces(self) -> int:
        return self.__len__()


class ContainerNPY(ContainerMemory):
    """Container loaded from .npy files via ``StorerMemory``."""

    @classmethod
    def from_dir(cls, path, fields=None, mmap_mode=None, frame=None, func_preprocess=None):
        """Create a memory container from .npy files in a directory."""
        path = Path(path)

        if fields is None:
            arrays = {
                npy_file.stem: _np.load(npy_file, mmap_mode=mmap_mode)
                for npy_file in sorted(path.glob("*.npy"))
            }
        else:
            arrays = {}
            for name, filename in fields.items():
                filename = Path(filename)
                npy_file = filename if filename.is_absolute() else path / filename
                arrays[name] = _np.load(npy_file, mmap_mode=mmap_mode)

        if "samples" not in arrays:
            raise FileNotFoundError("samples.npy is required")

        samples = arrays.pop("samples")
        # `poi` (sample positions, one value per leak point) describes the
        # trace layout, not per-trace data; keep it out of the tiled per-trace
        # metadata path and expose it as a plain `.poi` attribute instead.
        poi = arrays.pop("poi", None)
        if "key" in arrays and arrays["key"].ndim == 1:
            nb_traces = samples.reshape(-1, samples.shape[-1]).shape[0]
            arrays["key"] = _np.tile(arrays["key"], (nb_traces, 1))

        store = StorerMemory()
        store.update(samples=samples, **arrays)
        container = cls(store, frame=frame, func_preprocess=func_preprocess)
        if poi is not None:
            container.poi = poi
        return container
