# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import estraces as _ets
import numpy as _np
from nuscar.traceset import Container, Storer
import h5py as _h5py
import nuscar

class ReaderETS:
    def __init__(self, path: str):
        self.ths = _ets.read_ths_from_ets_file(path)

    def close(self):
        self.ths.close()

class StorerETS(Storer):
    def __init__(self, filename:str, mode='w'):
        super().__init__()
        self._storer = _h5py.File(filename, mode)
        self._is_first_update = True
        self._nb_cache_valid = 0
        if mode == 'a':
            self._is_first_write = False
        else:
            self._is_first_write = True        

    def _init_array(self, nb_samples, sdtype, **kwargs):
        """init arrays to cache a chunk.

        Args:
            nb_samples (int): number of samples.
            sdtype (dtype): dtype of sampels.
        """
        _chunk_size = nuscar._find_batch_size(nb_samples)
        self._samples = _np.zeros(
            (_chunk_size, nb_samples), dtype=sdtype)
        self._cache_meta = dict()
        for m in kwargs:
            d = kwargs[m]
            self._cache_meta[m] = _np.zeros(
                (_chunk_size, d.shape[-1]), dtype=d.dtype)
        self._nb_cache_valid = 0
        self._chunk_size = _chunk_size
        self._is_first_update = False
        

    def _init_ets_write(self):
        nb_traces = self._nb_cache_valid
        samples = self._samples[0:nb_traces].reshape(
            -1, self._samples.shape[-1])
        
        self._storer.create_dataset(
            'traces',  data=samples, compression="gzip", 
            chunks=True, maxshape=(None, samples.shape[-1]),
            dtype=samples.dtype
        )
        self._storer.create_group('metadata')

        
        for m in self._cache_meta:
            d = self._cache_meta[m][0:nb_traces]
            d = d.reshape(
                -1, d.shape[-1])
            self._storer['metadata'].create_dataset(
                m, data=d, compression="gzip", 
                chunks=True, maxshape=(None, d.shape[-1]),
                dtype=d.dtype
            )
        self._is_first_write = False

    def _update_chunk(self):
        if self._is_first_write:
            self._init_ets_write()
        else:
            nb_traces = self._nb_cache_valid
            samples = self._samples[0:nb_traces].reshape(
                -1, self._samples.shape[-1])
            self._storer['traces'].resize(self._storer['traces'].shape[0]+nb_traces, axis=0)
            self._storer['traces'][-nb_traces:] = samples
            
            for m in self._cache_meta:
                d = self._cache_meta[m][0:nb_traces]
                d = d.reshape(
                    -1, d.shape[-1])
                self._storer['metadata'][m].resize(self._storer['metadata'][m].shape[0]+nb_traces, axis=0)
                self._storer['metadata'][m][-nb_traces:] = d
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
                self._cache_meta[m][self._nb_cache_valid:,
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
                self._cache_meta[m][self._nb_cache_valid:self._nb_cache_valid+nb_traces,
                                   :] = d[idx:idx+nb_traces, :]
            self._nb_cache_valid = self._nb_cache_valid + nb_traces

        self.logger.debug("A batch recorded.")

    def close(self):
        if self._nb_cache_valid > 0:
            self._update_chunk()
        self._storer.close()

class ContainerETS(Container):
    """container generate from eshared ths format.
    """

    def __init__(self, reader, frame=None, func_preprocess=None):
        super().__init__(frame, func_preprocess)
        self._reader = reader
        self._sub_traceset_indices = None
        

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
        new_container = ContainerETS(
            self._reader, self._frame, self._func_pre)
        new_container._sub_traceset_indices = sub_traceset_indices
        return new_container

    def _convert_traces_indices_to_file_indices_array(self, traces):
        if self._sub_traceset_indices is not None:
            sub_max = len(self._sub_traceset_indices)
            traces_index = _np.array([t for t in traces if t < sub_max])
            return self._sub_traceset_indices[traces_index]
        else:
            sub_max = self.__len__()
            traces_index = _np.array([t for t in traces if t < sub_max])
            return _np.array(traces_index)


    def __len__(self):
        if self._sub_traceset_indices is not None:
            return len(self._sub_traceset_indices)
        return len(self._reader.ths)

    @ property
    def metadatas(self):
        return self._reader.ths.metadata_tags

    @property
    def _samples(self) -> _np.ndarray:
        if self._sub_traceset_indices is None:
            d = self._reader.ths.samples[:]
        else:
            d = self._reader.ths[self._sub_traceset_indices].samples[:]
        return d
    
    def __getattr__(self, name):
        try:
            assert (name in self.metadatas)
            if self._sub_traceset_indices is None:
                d = self._reader.ths.__getattr__(name)
            else:
                d = self._reader.ths[self._sub_traceset_indices].__getattr__(name)
            if len(d) == 1:
                return _np.squeeze(d)
            return d
        except:
            # or other errors that may occur
            raise AttributeError(name)

    @property
    def _nb_traces(self) -> int:
        return self.__len__()


class ReaderH5:
    def __init__(self, samples, **kwargs):
        self.ths = _ets.read_ths_from_ram(samples=samples, **kwargs)
        """
        import h5py
        file_name = "aes_hd.h5"
        f = h5py.File(file_name, 'r')
        
        for key in f.keys():
            print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
            
        group = f['Attack_traces']
        
        #Checkout what keys are inside that group.
        for key in group.keys():
            print(key)
        
        # This assumes group[some_key_inside_the_group] is a dataset, 
        # and returns a np.array:
        traces = group['traces'][()]
        metadata = group['metadata'][()]
        ma = {'plaintext': np.array([x[0] for x in metadata]), 
              'ciphertext': np.array([x[1] for x in metadata]), 
              'key': np.array([x[2] for x in metadata])}
        reader = nuscar.traceset.ReaderH5(samples=traces, plaintext=ma['plaintext'])
        """

    def close(self):
        self.ths.close()


class ReaderTRS:
    def __init__(self, path: str, meta: dict, datatype=None):
        self.ths = _ets.read_ths_from_trs_file(filename=path, metadatas_parsers=meta, dtype=datatype)
        """
        metadatas_parsers=dict(plaintext=lambda x: np.frombuffer(x, dtype='uint8', offset=0, count=16),
                               ciphertext=lambda x: np.frombuffer(x, dtype='uint8', offset=16))
        """


    def close(self):
        self.ths.close()

