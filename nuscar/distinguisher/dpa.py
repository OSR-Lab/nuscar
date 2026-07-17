# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
from nuscar.distinguisher import DistinguisherBase


class DPADistinguisher(DistinguisherBase):
    """Differential Power Analysis
    """

    def __init__(self, name="DPA", precision: str = 'f64'):
        super().__init__(name, precision)

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            nb_samples = samples.shape[1]
            nb_data = data.shape[1]
            self.acc_traces = _np.zeros(nb_samples, dtype=self.precision)
            self.processed_ones = _np.zeros(nb_data, dtype=self.precision)
            self.acc_ones = _np.zeros((nb_data, nb_samples), dtype=self.precision)
            self.is_initialized = True

        _samples = samples.astype(self.precision, copy=False)
        _data = data.astype(self.precision, copy=False)
        self.processed_ones += _np.sum(_data, axis=0)
        self.acc_ones += _np.dot(_data.T, _samples)
        self.acc_traces += _np.sum(_samples, axis=0)
        super().update(samples, data)

    def final(self) -> _np.ndarray:
        super().final()
        acc_1 = (self.acc_ones.swapaxes(0, 1) /
                 self.processed_ones).swapaxes(0, 1)

        processed_zeros = self._processed_traces-self.processed_ones
        acc_zeros = (self.acc_traces-self.acc_ones)
        acc_0 = (acc_zeros.swapaxes(0, 1) /
                 processed_zeros).swapaxes(0, 1)
        return acc_1-acc_0
