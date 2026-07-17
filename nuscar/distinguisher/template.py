# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar.nuscar_rust as _n_rust
import zarr as _zarr
import logging

from nuscar.distinguisher import DistinguisherBase


class TemplateModel():
    """Template Model"""

    def __init__(self, means: _np.ndarray = None, pooled_cov_inv: _np.ndarray = None):
        self.pooled_cov_inv = pooled_cov_inv
        self.means = means

    def save(self, path):
        """Save the model to a zip file.
        """
        if self.pooled_cov_inv is None or self.means is None:
            raise RuntimeError("Current model is not initialized, cannot save")
        _storer = _zarr.storage.ZipStore(path, mode='w')
        _g_root = _zarr.group(store=_storer, overwrite=True)
        _g_root['means'] = self.means
        _g_root['pcovs_inv'] = self.pooled_cov_inv
        _storer.close()

    def load(self, path):
        """Load the model from a zip file.
        """
        _storer = _zarr.storage.ZipStore(path, read_only=False)
        _g_root = _zarr.group(store=_storer)
        self.means = _g_root['means'][:]
        self.pooled_cov_inv = _g_root['pcovs_inv'][:]
        _storer.close()


class TemplateTrain(DistinguisherBase):
    """Train a template model.
    """

    def __init__(self, nb_labels: int, name="Train Template", precision: str = 'f64'):
        """Init a template model train.

        Args:
            nb_labels (int): The number of labels indicates how many classes template model take.
            For example if we use Hamming Weight, the labels will be 9 for a single byte.
            The output of selection function must match this number.
        """
        super().__init__(name, precision)
        self.nb_labels = nb_labels
        self.logger = logging.getLogger(__name__)

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            nb_samples = samples.shape[1]
            self._x = _np.zeros(
                (self.nb_labels, nb_samples), dtype=self.precision)
            self._cov = _np.zeros(
                (self.nb_labels, nb_samples, nb_samples), dtype=self.precision)
            self.counter = _np.zeros(self.nb_labels, dtype=_np.uint32)
            self.is_initialized = True

        _samples = samples.astype(self.precision, copy=False)
        _data = data.astype(_np.uint8, copy=False)
        # use rust to accelerate
        if self.precision == _np.float64:
            _n_rust.template_update_train_r(
                self._pool, self._x, self._cov, self.counter, _samples, _data)
        else:
            _n_rust.template_update_train_r32(
                self._pool, self._x, self._cov, self.counter, _samples, _data)
        super().update(samples, data)

    def final(self):
        super().final()
        self.means = (self._x.swapaxes(0, 1) / self.counter).swapaxes(0, 1)
        self.pooled_cov = _np.zeros(self._cov.shape[1:], dtype=self.precision)
        for i in range(self.nb_labels):
            tmp = _np.outer(self.means[i], self.means[i]) * self.counter[i]
            self.pooled_cov += (self._cov[i] - tmp) / (self.counter[i] - 1)
        bad_class_idx = []
        for i, c in enumerate(self.counter):
            if c < 2:
                bad_class_idx.append(i)
        if len(bad_class_idx) > 0:
            self.logger.error(f"Class %s has insufficient samples" % str(bad_class_idx))
            return
        self.pooled_cov /= self.nb_labels
        self.pooled_cov_inv = _np.linalg.pinv(self.pooled_cov)

        return None

    @ property
    def model(self):
        return TemplateModel(self.means, self.pooled_cov_inv)


class TemplateApply(DistinguisherBase):
    """Apply data with trainned template.
    """

    def __init__(self, model: TemplateModel, name="Template", precision: str = 'f64'):
        """Init a template model fit. 
        """
        super().__init__(name, precision)
        if model.means is None or model.pooled_cov_inv is None:
            raise ValueError("Template model is invalid!")
        self.model = model

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            if data is None:
                self.is_dpa_template = False
                self.scores = _np.zeros(
                    self.model.means.shape[0], dtype=self.precision)
            else:
                self.is_dpa_template = True
                self.scores = _np.zeros(
                    data.shape[1], dtype=self.precision)
                self.nb_data = data.shape[1]
            self.is_initialized = True
        assert (samples.shape[1] == self.model.means.shape[1])

        if self.is_dpa_template:
            for i in range(self.nb_data):
                tmp = self.model.means[data[:, i]]  # nb_traces, nb_samples
                tmp_mean = samples - tmp            # nb_traces, nb_samples
                tmp = _np.dot(tmp_mean, self.model.pooled_cov_inv)
                tmp = tmp_mean * tmp
                self.scores[i] += _np.sum(tmp) / samples.shape[1]
        else:
            for i in range(len(self.scores)):
                # nb_traces, nb_samples
                tmp_mean = samples - self.model.means[i]
                tmp = _np.dot(tmp_mean, self.model.pooled_cov_inv)
                tmp = tmp_mean * tmp
                self.scores[i] += _np.sum(tmp) / samples.shape[1]

        super().update(samples, data)

    def final(self):
        super().final()
        if self.is_dpa_template:
            # expand to 3D to meet display requirements
            return (10-(self.scores/self._processed_traces))[:, _np.newaxis, _np.newaxis]
        else:
            # expand to 2D
            return (10-(self.scores/self._processed_traces))[_np.newaxis, :]
