# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

import numpy as _np
import logging
from sklearn.linear_model import SGDClassifier as _SGDClassifier
from sklearn.linear_model import Perceptron as _Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier as _PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
from nuscar.distinguisher import DistinguisherBase
from nuscar.distinguisher.base import MLModel


class SGDTrain(DistinguisherBase):
    """Train a sgd model with SGDClassifier.
    """

    def __init__(self, nb_labels: int, name="Train SGD", precision: str = 'f64'):
        """Init a SGD model train.
        """
        super().__init__(name, precision)
        self.logger = logging.getLogger(__name__)
        self.classes = list(range(nb_labels))

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            self.clf = _SGDClassifier()
            self.is_initialized = True

        X = samples.astype(self.precision, copy=False)
        Y = _np.squeeze(data.astype(_np.uint8, copy=False))
        self.clf.partial_fit(X, Y, classes=self.classes)
        super().update(samples, data)

    def final(self):
        super().final()

        return None

    @ property
    def model(self):
        return MLModel(self.clf, name='SGDClassifier')


class MultinomialNBTrain(DistinguisherBase):
    """Train a MultinomialNB model with MultinomialNB.
    """

    def __init__(self, nb_labels: int, name="Train MultinomialNB", precision: str = 'f64'):
        """Init a MultinomialNB model train.
        """
        super().__init__(name, precision)
        self.logger = logging.getLogger(__name__)
        self.classes = list(range(nb_labels))

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            self.clf = _MultinomialNB()
            self.is_initialized = True

        X = samples.astype(self.precision, copy=False)
        Y = _np.squeeze(data.astype(_np.uint8, copy=False))
        self.clf.partial_fit(X, Y, classes=self.classes)
        super().update(samples, data)

    def final(self):
        super().final()

        return None

    @ property
    def model(self):
        return MLModel(self.clf, name='MultinomialNB')


class PerceptronTrain(DistinguisherBase):
    """Train a Perceptron model.
    """

    def __init__(self, nb_labels: int, name="Train Perceptron", precision: str = 'f64'):
        """Init a Perceptron model train.
        """
        super().__init__(name, precision)
        self.logger = logging.getLogger(__name__)
        self.classes = list(range(nb_labels))

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            self.clf = _Perceptron()
            self.is_initialized = True

        X = samples.astype(self.precision, copy=False)
        Y = _np.squeeze(data.astype(_np.uint8, copy=False))
        self.clf.partial_fit(X, Y, classes=self.classes)
        super().update(samples, data)

    def final(self):
        super().final()

        return None

    @ property
    def model(self):
        return MLModel(self.clf, name='Perceptron')


class PassiveAggressiveTrain(DistinguisherBase):
    """Train a PassiveAggressiveClassifier model.
    """

    def __init__(self, nb_labels: int, name="Train PassiveAggressiveClassifier", precision: str = 'f64'):
        """Init a PassiveAggressiveClassifier model train.
        """
        super().__init__(name, precision)
        self.logger = logging.getLogger(__name__)
        self.classes = list(range(nb_labels))

    def update(self, samples: _np.ndarray, data: _np.ndarray):
        if not self.is_initialized:
            self.clf = _PassiveAggressiveClassifier()
            self.is_initialized = True

        X = samples.astype(self.precision, copy=False)
        Y = _np.squeeze(data.astype(_np.uint8, copy=False))
        self.clf.partial_fit(X, Y, classes=self.classes)
        super().update(samples, data)

    def final(self):
        super().final()

        return None

    @ property
    def model(self):
        return MLModel(self.clf, name='PassiveAggressive')
