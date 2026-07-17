# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
from nuscar.distinguisher import DistinguisherBase


class TTestDistinguisher(DistinguisherBase):
    """TTest leakage detection
    """

    def __init__(self, name="TTest", precision: str = 'f64'):
        super().__init__(name, precision)

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            nb_samples = samples.shape[1]
            self.ex = _np.zeros(nb_samples, dtype=self.precision)
            self.ex2 = _np.zeros(nb_samples, dtype=self.precision)
            self.ey = _np.zeros(nb_samples, dtype=self.precision)
            self.ey2 = _np.zeros(nb_samples, dtype=self.precision)
            self.num_x = 0
            self.num_y = 0
            self.is_initialized = True

        _samples = samples.astype(self.precision, copy=False)
        _data = data.astype(_np.uint32, copy=False)
        # use rust to accelerate
        if self.precision == _np.float64:
            nx, ny = _n_rust.ttest_update_r(self._pool, self.ex, self.ex2, self.ey,
                                            self.ey2, _samples, _data)
        else:
            nx, ny = _n_rust.ttest_update_r32(self._pool, self.ex, self.ex2, self.ey,
                                              self.ey2, _samples, _data)

        super().update(samples, data)
        self.num_x = nx+self.num_x
        self.num_y = ny+self.num_y

    def final(self) -> _np.ndarray:
        super().final()
        # use rust to accelerate
        if self.precision == _np.float64:
            return _n_rust.ttest_final_r(self._pool, self.ex, self.ex2, self.ey,
                                         self.ey2, self.num_x, self.num_y)
        else:
            return _n_rust.ttest_final_r32(self._pool, self.ex, self.ex2, self.ey,
                                           self.ey2, self.num_x, self.num_y)
