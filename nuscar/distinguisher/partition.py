# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from nuscar.distinguisher import DistinguisherBase
import logging


class ANOVADistinguisher(DistinguisherBase):
    """ANOVA Analysis
    """

    def __init__(self, partitions, name="ANOVA", precision: str = 'f64'):
        """

        Args:
            partitions (ndarray): partitions to categorize traces. e.g: np.arange(9) for Hamming Weight model on bytes.
        """
        super().__init__(name, precision)
        self.partitions = partitions
        self.logger = logging.getLogger(__name__)

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        nb_samples = samples.shape[1]
        nb_data = data.shape[1]
        n_partitions = len(self.partitions)
        if not self.is_initialized:
            self.ex = _np.zeros(
                shape=(nb_data, nb_samples, n_partitions), dtype=self.precision)
            self.ex2 = _np.zeros(
                shape=(nb_data, nb_samples, n_partitions), dtype=self.precision)
            self.counters = _np.zeros(
                shape=(nb_data, len(self.partitions)), dtype=_np.uint32)
            self.is_initialized = True

        _samples = samples.astype(self.precision, copy=False)
        _data = data.astype(_np.uint8, copy=False)
        # use rust to accelerate
        if self.precision == _np.float64:
            _n_rust.partition_update_r(
                self._pool, self.ex, self.ex2, self.counters, _samples, _data)
        else:
            _n_rust.partition_update_r32(
                self._pool, self.ex, self.ex2, self.counters, _samples, _data)
        super().update(samples, data)

    def final(self) -> _np.ndarray:
        super().final()
        nb_data = self.ex.shape[0]
        nb_samples = self.ex.shape[1]

        result = _np.empty((nb_data, nb_samples), dtype=self.precision)
        for i in range(nb_data):
            idx = self.counters[i] > 0
            ccounters = self.counters[i, idx]
            EE1 = self.ex[i][:, idx]
            EE2 = self.ex2[i][:, idx]
            N = _np.sum(ccounters)
            MEE1 = _np.sum(EE1, axis=1) / N
            numerator = (((EE1/ccounters).T - MEE1).T)**2 * ccounters
            numerator = _np.sum(numerator, axis=1)
            numerator /= idx.shape[0]
            denominator = _np.sum(EE2, axis=1)/N - (MEE1)**2

            tmp_result = numerator / denominator / N
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            result[i] = tmp_result
        return result


class NICVDistinguisher(ANOVADistinguisher):
    """NICV Analysis
    """

    def __init__(self, partitions, name="NICV", precision: str = 'f64'):
        """

        Args:
            partitions (ndarray): partitions to categorize traces. e.g: np.arange(9) for Hamming Weight model on bytes.
        """
        super().__init__(partitions, name, precision)


    def final(self) -> _np.ndarray:
        super().final()
        nb_data = self.ex.shape[0]
        nb_samples = self.ex.shape[1]

        result = _np.empty((nb_data, nb_samples), dtype=self.precision)
        for i in range(nb_data):
            idx = self.counters[i] > 0
            ccounters = self.counters[i, idx]
            EE1 = self.ex[i][:, idx]
            EE2 = self.ex2[i][:, idx]
            N = _np.sum(ccounters)
            MEE1 = _np.sum(EE1, axis=1) / N

            numerator = (((EE1/ccounters).T - MEE1).T)**2
            numerator *= ccounters/N
            numerator = _np.sum(numerator, axis=1)

            denominator = _np.sum(EE2, axis=1)/N - (MEE1)**2

            tmp_result = numerator / denominator
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            result[i] = tmp_result
        return result


class SNRDistinguisher(ANOVADistinguisher):
    """SNR Analysis
    """

    def __init__(self, partitions, name="SNR", precision: str = 'f64'):
        """

        Args:
            partitions (ndarray): partitions to categorize traces. e.g: np.arange(9) for Hamming Weight model on bytes.
        """
        super().__init__(partitions, name, precision)

    def final(self) -> _np.ndarray:
        super().final()
        nb_data = self.ex.shape[0]
        nb_samples = self.ex.shape[1]

        result = _np.empty((nb_data, nb_samples), dtype=self.precision)
        for i in range(nb_data):
            idx = self.counters[i] > 0
            ccounters = self.counters[i, idx]
            EE1 = self.ex[i][:, idx]
            EE2 = self.ex2[i][:, idx]
            N = _np.sum(ccounters)
            MEE1 = _np.sum(EE1, axis=1) / N
            numerator = (((EE1/ccounters).T - MEE1).T)**2
            numerator = _np.sum(numerator, axis=1)
            numerator /= idx.shape[0]

            denominator = (EE2/ccounters) - (EE1/ccounters)**2
            denominator = _np.sum(denominator, axis=1)
            denominator /= idx.shape[0]

            tmp_result = numerator / denominator
            tmp_result[_np.isinf(tmp_result)] = _np.nan
            result[i] = tmp_result
        return result
