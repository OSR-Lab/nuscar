# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import nuscar
import joblib as _joblib


class DistinguisherBase():
    def __init__(self, name: str = "base", precision: str = 'f64'):
        self._is_init = False
        self._processed_traces = 0
        if precision == 'f64':
            self.precision = _np.float64
        elif precision == 'f32':
            self.precision = _np.float32
        else:
            raise ValueError("precision must be 'f64' or 'f32'")
        self._pool = nuscar._global_pool
        self.name = name

    @property
    def is_initialized(self):
        return self._is_init

    @is_initialized.setter
    def is_initialized(self, yn: bool):
        if self._is_init is True and yn is False:
            self._processed_traces = 0  # reset to 0
        self._is_init = yn

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        """update internal value with fresh samples and data.

        Args:
            samples (_np.ndarray): with shape (batch_size, nb_samples)
            data (_np.ndarray): with shape (batch_size, nb_data)
        """
        self._processed_traces = self._processed_traces + samples.shape[0]

    def final(self) -> _np.ndarray:
        """calculate distinguisher result from internal value.

        Returns:
            _np.ndarray: with shape (m, nb_samples), e.g. m = nb_data * guesses
        """
        if not self._is_init:
            raise RuntimeError("could not final without update")


class MLModel():
    """machine learning model"""

    def __init__(self, clf=None, name=None):
        self.clf = clf
        self.name = name

    def save(self, path):
        """Save the model
        """
        if self.clf is None:
            raise RuntimeError("Current model is not initialized, cannot save")
        _joblib.dump(self.clf, path, compress=9)

    def load(self, path):
        """Load the model from a file.
        """
        self.clf = _joblib.load(path)


class MLApply(DistinguisherBase):
    """Match data with trainned model.
    """

    def __init__(self, model: MLModel, name="ML", precision: str = 'f64'):
        """Init a model fit.
        """
        super().__init__(name, precision)
        if model.clf is None:
            raise ValueError("Model is invalid!")
        self.name = 'Matchine Learning '+model.name
        self.model = model

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            if data is None:
                self.is_dpa_template = False
                self.scores = _np.zeros_like(
                    self.model.clf.classes_, dtype=self.precision)
            else:
                self.is_dpa_template = True
                self.scores = _np.zeros(
                    data.shape[1], dtype=self.precision)
                self.nb_data = data.shape[1]
            self.is_initialized = True
        X = samples.astype(self.precision, copy=False)
        if self.is_dpa_template:
            for i in range(self.nb_data):
                self.scores[i] += self.model.clf.score(X, data[:, i])
        else:
            for i in range(len(self.scores)):
                Y = _np.array([i]*X.shape[0], dtype=_np.uint8)
                self.scores[i] += self.model.clf.score(X, Y)
        super().update(samples, data)

    def final(self):
        super().final()
        if self.is_dpa_template:
            # expand to 3D to meet display requirements
            return self.scores[:, _np.newaxis, _np.newaxis]
        else:
            # expand to 2D
            return self.scores[_np.newaxis, :]
