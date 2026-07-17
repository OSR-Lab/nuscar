# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import numba as _nb
from nuscar.distinguisher import DistinguisherBase

def _set_histogram_parameters(obj, bins_number, bin_edges):
    #print("_set_histogram_parameters")
    if not isinstance(bins_number, int):
        raise TypeError(f'bins_number must be an integer, not {type(bins_number)}.')
    obj.bins_number = bins_number
    obj._bin_edges = None
    obj.y_window = None
    if bin_edges is not None:
        obj.bin_edges = bin_edges

class MIADistinguisher(DistinguisherBase):
    """This partitioned distinguisher mixin applies a mutual information computation."""

    def __init__(self, bins_number=128, bin_edges=None, name="MIA", precision: str = 'f32'):
        super().__init__(name, precision)
        _set_histogram_parameters(self, bins_number=bins_number, bin_edges=bin_edges)
        self.partitions = None

    def _init_partitions(self, data):
        maxdata = _np.nanmax(data)
        mindata = _np.nanmin(data)
        if self.partitions is None:
            if maxdata > 255:
                raise ValueError('max value for intermediate data is greater than 255, you need to provide partitions explicitly at init.')
            if mindata < 0:
                raise ValueError('min value for intermediate data is lower than 0, you need to provide partitions explicitly at init.')
            ls = [0, 9, 64, 256]
            for r in ls:
                if maxdata <= r:
                    break
            self.partitions = _np.arange(r, dtype='int32')
    def _memory_usage(self, samples, data):
        #print("_memory_usage")
        self._init_partitions(data)
        self._init_bin_edges(samples)
        dtype_size = _np.dtype(self.precision).itemsize
        return 3 * dtype_size * data.shape[1] * samples.shape[1] * len(self.partitions) * self.bins_number

    def _init_bin_edges(self, samples):
        #print("_init_bin_edges")
        if self.bin_edges is None:
            self.y_window = (_np.min(samples), _np.max(samples))
            self.bin_edges = _np.linspace(*self.y_window, self.bins_number + 1)

    @property
    def bin_edges(self):
        #print("bin_edges 1")
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, bin_edges):
        #print("bin_edges 2")
        if bin_edges is None or not isinstance(bin_edges, (list, _np.ndarray, range)):
            raise TypeError(f'bin_edges must be a ndarray, a list or a range, not {type(bin_edges)}.')
        if len(bin_edges) <= 1:
            raise ValueError(f'bin_edges length must be >1, but {len(bin_edges)}, found.')
        if not isinstance(bin_edges, _np.ndarray):
            bin_edges = _np.array(bin_edges, dtype=self.precision)
        for a, b in zip(bin_edges, bin_edges[1:]):
            if not a < b:
                raise ValueError(f'bin_edges must be sorted, but {a} >= {b}.')
        if _np.sum(_np.diff(_np.diff(bin_edges))) > 1e-9:
            raise ValueError('bin_edges must be uniform (i.e with bins equally spaced.')
        self._bin_edges = bin_edges
        self.bins_number = len(bin_edges) - 1


    def _initialize_accumulators(self):
        #print("_initialize_accumulators",self._data_words)
        self.accumulators = _np.zeros((self._trace_length, self.bins_number, len(self.partitions), self._data_words),
                                      dtype=self.precision)

    @staticmethod
    #@_nb.njit(parallel=True)
    def _accumulate_core(samples, data, self_bin_edges, self_accumulators):
        #print("_accumulate_core")
        nbins = len(self_bin_edges) - 1
        min_edge = self_bin_edges[0]
        max_edge = self_bin_edges[-1]
        norm = nbins / (max_edge - min_edge)

        for sample_idx in _nb.prange(samples.shape[1]):
            for trace_idx in range(samples.shape[0]):
                x = samples[trace_idx, sample_idx]
                if x >= min_edge and x < max_edge:
                    bin_idx = int((x - min_edge) * norm)
                elif x == max_edge:
                    bin_idx = nbins - 1
                else:
                    continue
                for data_idx in range(data.shape[1]):
                    self_accumulators[sample_idx, bin_idx, data[trace_idx, data_idx], data_idx] += 1

    def _accumulate(self, samples, data):
        #print("_accumulate",data)
        self._accumulate_core(samples, data, self.bin_edges, self.accumulators)

    def _compute_pdf(self, array, axis):
        #print("_compute_pdf")
        s = array.sum(axis=axis)
        s[s == 0] = 1
        return (array.swapaxes(0, 1) / s).swapaxes(0, 1)

    def _compute(self):
        #print("_compute")
        background = self.accumulators.sum(axis=2)

        pdfs_background = self._compute_pdf(background, axis=1)
        pdfs_background[pdfs_background == 0] = 1

        pdfs_of_histos = self._compute_pdf(self.accumulators, axis=1)
        pdfs_of_histos[pdfs_of_histos == 0] = 1

        histos_sums = self.accumulators.sum(axis=1)
        ratios = (histos_sums.swapaxes(0, 1) / background.sum(axis=1)).swapaxes(0, 1)
        expected = pdfs_background * _np.log(pdfs_background)
        real = pdfs_of_histos * _np.log(pdfs_of_histos)
        delta = (real.swapaxes(1, 2).swapaxes(0, 1) - expected).swapaxes(0, 1).swapaxes(1, 2)
        res = delta.sum(axis=1) * ratios
        return _np.sum(res, axis=1).swapaxes(0, 1)
        
    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            self._trace_length = samples.shape[1]
            self._data_words = data.shape[1]
            self._memory_usage(samples, data)
            self._initialize_accumulators()
            self.is_initialized = True

        # TODO: do we need to assert shape of samples and data match internal value here?
        self._accumulate(samples, data)

        super().update(samples, data)

    def final(self) -> _np.ndarray:
        #print("MIA Over")
        super().final()
        result=self._compute()
        #_np.save('data1.npy',result)
        
        return result  # use rust to accelerate
