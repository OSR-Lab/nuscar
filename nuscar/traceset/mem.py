# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
from nuscar.traceset import Container
from nuscar.traceset import Storer


class StorerMemory(Storer):
    def __init__(self):
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
        super().update(samples, **kwargs)
        if self._is_first_update:
            self._init_array(samples, **kwargs)
        else:
            self._samples = _np.vstack((self._samples, samples))
            for m in kwargs:
                d = kwargs[m]
                self._mem_meta[m] = _np.vstack((self._mem_meta[m], d))
        
    def close(self):
        pass


class ContainerMemory(Container):
    def __init__(self, memstore:StorerMemory, frame=None, func_preprocess=None):
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
