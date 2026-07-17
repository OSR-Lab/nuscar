# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from nuscar.distinguisher import DistinguisherBase
import logging


class LRADistinguisher(DistinguisherBase):
    """Linear Regression Analysis
    (M. Ouladj, S. Guilley, E. Prouff, On the Implementation Efficiency of Linear Regression-Based Side-Channel Attacks)
    """

    def __init__(self,  T: _np.ndarray, guesses=None, partitions=256, alpha=1e20, name="LRA", precision: str = 'f64'):
        """

        Args:
            T (ndarray): preprocessed 1-D/2-D array. (nb_input_value) or (nb_key_guesses, nb_input_value). 
                For all possible input value, calculate the possible target.
            alpha (float): lra result will be close to 1.0. alpha will compute the result as r = alpha^r/alpha to increase distinction.

        """
        super().__init__(name, precision)
        bit_coeff = {'uint8': 9, 'uint16': 17, 'uint32': 33}
        if isinstance(partitions, int):
            self._partition_range = range(partitions)
        else:
            self._partition_range = partitions
        self.guesses = guesses
        if guesses is not None:
            if not len(guesses) == T.shape[0]:
                raise ValueError("guesses length must match the length of input T")
        if len(T.shape) == 2:
            self._nb_guesses = T.shape[0]
            self._nb_all_input = T.shape[1]  # iterate over all inputs
        else:
            self._nb_guesses = 1  # leakage mode, not attack
            self._nb_all_input = T.shape[0]
        if not str(T.dtype) in bit_coeff:
            raise TypeError('T type only supports uint8, uint16 and uint32')
        self._nb_coeff = bit_coeff[str(T.dtype)]

        T = T.reshape((-1, self._nb_all_input))
        self.P = _np.zeros((self._nb_guesses, self._nb_coeff,
                           self._nb_all_input), dtype=self.precision)
        self.M = _np.zeros(
            (self._nb_guesses, self._nb_all_input, self._nb_coeff), dtype=self.precision)
        self.M[:, :, 0] = 1
        for i in range(1, self._nb_coeff):
            self.M[:, :, i] = (T >> (self._nb_coeff-1-i)) & 1
        for k in range(self._nb_guesses):
            self.P[k] = _np.linalg.inv(self.M[k].T @ self.M[k]) @ self.M[k].T
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        nb_samples = samples.shape[1]
        nb_data = data.shape[1]
        n_partitions = len(self._partition_range)
        if not self.is_initialized:
            self.ex = _np.zeros(
                shape=(nb_data, nb_samples, n_partitions), dtype=self.precision)
            self.ex2 = _np.zeros(
                shape=(nb_data, nb_samples, n_partitions), dtype=self.precision)
            self.counters = _np.zeros(
                shape=(nb_data, n_partitions), dtype=_np.uint32)
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
        u = self.ex.sum(axis=2)  # nb_data * nb_samples
        v = self.ex2.sum(axis=2)  # nb_data * nb_samples
        SST = v - (u ** 2) / self.counters.sum(axis=1)[:, _np.newaxis]
        # nb_data * nb_samples * n_partitions
        L = _np.nan_to_num(self.ex/self.counters[:, _np.newaxis, :])

        result = _np.empty(
            (self._nb_guesses, nb_data, nb_samples), dtype=self.precision)
        for i in range(nb_data):
            Li = L[i].T
            for k in range(self._nb_guesses):
                beta = self.P[k] @ Li
                epsilon = self.M[k] @ beta
                SSR = _np.sum((epsilon - Li)**2, axis=0)
                result[k, i, :] = self.alpha**(1 - (SSR/SST[i, :]))/self.alpha
        return result.reshape((-1, nb_samples))  # adapt for task
