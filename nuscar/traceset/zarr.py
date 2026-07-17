# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import zarr as _zarr
import nuscar
from nuscar.traceset import Container
from nuscar.traceset import Storer
import enum

class StorerZARR(Storer):
    def __init__(self, filename: str, mode = 'w'):
        super().__init__()
        
        import warnings
        warnings.filterwarnings("ignore", 
                        category=UserWarning, 
                        message="Duplicate name.*")
        
        self._storer = _zarr.storage.ZipStore(filename, mode=mode, read_only=False)
       
        self._is_first_update = True
        
        if mode == 'a':
            self._g_root = _zarr.open_group(store=self._storer)
            self._g_meta = _zarr.open_group(store=self._storer, path='meta')
            self._is_first_write = False
        else:
            self._g_root = _zarr.group(store=self._storer, overwrite=True)
            self._g_meta = self._g_root.create_group('meta')
            self._is_first_write = True
            
    def _init_array(self, nb_samples, sdtype, **kwargs):
        _chunk_size = nuscar._find_batch_size(nb_samples)
        self._samples = _np.zeros(
            (_chunk_size, nb_samples), dtype=sdtype)
        self._zarr_meta = dict()
        for m in kwargs:
            d = kwargs[m]
            self._zarr_meta[m] = _np.zeros(
                (_chunk_size, d.shape[-1]), dtype=d.dtype)
        self._nb_cache_valid = 0
        self._chunk_size = _chunk_size
        self._is_first_update = False

    def _init_zarr_write(self):
        nb_traces = self._nb_cache_valid
        samples = self._samples[0:nb_traces].reshape(
            -1, self._samples.shape[-1])
        self._g_root['samples'] = samples
        for m in self._zarr_meta:
            d = self._zarr_meta[m][0:nb_traces]
            d = d.reshape(
                -1, d.shape[-1])
            self._g_meta[m] = d
        self._is_first_write = False

    def _update_chunk(self):
        if self._is_first_write:
            self._init_zarr_write()
        else:
            nb_traces = self._nb_cache_valid
            samples = self._samples[0:nb_traces].reshape(
                -1, self._samples.shape[-1])
            self._g_root['samples'].append(samples)
            for m in self._zarr_meta:
                d = self._zarr_meta[m][0:nb_traces]
                d = d.reshape(
                    -1, d.shape[-1])
                self._g_meta[m].append(d)
        self._nb_cache_valid = 0

    def update(self, samples: _np.array, **kwargs):
        super().update(samples, **kwargs)
        samples = samples.reshape(-1, samples.shape[-1])
        nb_traces = samples.shape[0]
        if self._is_first_update:
            self._init_array(samples.shape[-1], samples.dtype, **kwargs)

        nb_cache_left = self._chunk_size - self._nb_cache_valid
        idx = 0
        while nb_traces > nb_cache_left:
            self._samples[self._nb_cache_valid:,
                          :] = samples[idx:idx+nb_cache_left, :]
            for m in kwargs:
                d = kwargs[m]
                d = d.reshape(-1, d.shape[-1])
                self._zarr_meta[m][self._nb_cache_valid:,
                                   :] = d[idx:idx+nb_cache_left, :]
            self._nb_cache_valid = self._nb_cache_valid + nb_cache_left
            self._update_chunk()
            idx = idx + nb_cache_left
            nb_traces = nb_traces-nb_cache_left
            nb_cache_left = self._chunk_size

        if nb_traces > 0:
            self._samples[self._nb_cache_valid:self._nb_cache_valid+nb_traces,
                          :] = samples[idx:idx+nb_traces, :]
            for m in kwargs:
                d = kwargs[m]
                d = d.reshape(-1, d.shape[-1])
                self._zarr_meta[m][self._nb_cache_valid:self._nb_cache_valid+nb_traces,
                                   :] = d[idx:idx+nb_traces, :]
            self._nb_cache_valid = self._nb_cache_valid + nb_traces

        self.logger.debug("A batch recorded.")

    def close(self):
        if self._nb_cache_valid > 0:
            self._update_chunk()
        self._storer.close()


class ReaderZARR():
    def __init__(self, path: str):
        self._storer = _zarr.storage.ZipStore(path, read_only=False)
    def close(self):
        self._storer.close()


class ContainerZARR(Container):
    def __init__(self, reader: ReaderZARR, frame=None, func_preprocess=None):
        super().__init__(frame, func_preprocess)
        self._reader = reader
        self._sub_traceset_indices = None
        self._g_root = _zarr.group(store=self._reader._storer)
        self._g_meta = self._g_root['meta']

    def __len__(self):
        if self._sub_traceset_indices is not None:
            return len(self._sub_traceset_indices)
        return self._g_root['samples'].shape[0]

    @ property
    def metadatas(self):
        return list(self._g_meta.array_keys())

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
        new_container = ContainerZARR(
            self._reader, self._frame, self._func_pre)
        new_container._sub_traceset_indices = sub_traceset_indices
        return new_container

    def __getattr__(self, name):
        try:
            assert (name in self.metadatas)
            if self._sub_traceset_indices is None:
                d = self._g_meta[name][:]
            else:
                d = self._g_meta[name].get_orthogonal_selection(
                    (self._sub_traceset_indices, slice(None)))
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
            d = self._g_root['samples'][:]
        else:
            d = self._g_root['samples'].get_orthogonal_selection(
                (self._sub_traceset_indices, slice(None)))
        return d

    @property
    def _nb_traces(self) -> int:
        return self.__len__()
