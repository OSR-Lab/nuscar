# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from nuscar.distinguisher import DistinguisherBase


class CPADistinguisher(DistinguisherBase):
    """Correlation Power Analysis
    """

    def __init__(self, name="CPA", precision: str = 'f64'):
        super().__init__(name, precision)
        
    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            nb_samples = samples.shape[1]
            nb_data = data.shape[1]
            self.ex = _np.zeros(nb_samples, dtype=self.precision)
            self.ex2 = _np.zeros(nb_samples, dtype=self.precision)
            self.ey = _np.zeros(nb_data, dtype=self.precision)
            self.ey2 = _np.zeros(nb_data, dtype=self.precision)
            self.exy = _np.zeros((nb_data, nb_samples), dtype=self.precision)
            self.is_initialized = True

        # TODO: do we need to assert shape of samples and data match internal value here?
        _samples = samples.astype(self.precision, copy=False)
        _data = data.astype(self.precision, copy=False)
        if self.precision == _np.float64:
            _n_rust.cpa_update_r(self._pool, self.ex, self.ex2, self.ey,
                                 self.ey2, self.exy, _samples, _data)  # use rust to accelerate
        else:
            _n_rust.cpa_update_r32(self._pool, self.ex, self.ex2, self.ey,
                                 self.ey2, self.exy, _samples, _data)  # use rust to accelerate
        super().update(samples, data)

    def final(self) -> _np.ndarray:
        super().final()
        if self.precision == _np.float64:
            return _n_rust.cpa_final_r(self._pool, self.ex, self.ex2, self.ey,
                                   self.ey2, self.exy, self._processed_traces)  # use rust to accelerate
        else:
            return _n_rust.cpa_final_r32(self._pool, self.ex, self.ex2, self.ey,
                                   self.ey2, self.exy, self._processed_traces)  # use rust to accelerate
