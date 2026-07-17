# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

"""High performance lib for side channel attacks."""
import psutil
import logging.handlers

from .view import (
    plot,
    plot_multi,
    plot_separate,
    plot_seperate,
    plot_subfigure,
    plot_pattern,
    plot_peak,
    plot_heatmap,
    plot_spectrogram
)

from .nuscar_rust import ThreadPool as _ThreadPool

from . import signalproc
from . import leakmodel
from . import ciphers
from . import highorder
from . import collect
from . import traceset

# register_plotly_resampler(mode='auto')

# from .distinguisher import (
#     CPADistinguisher,
#     TTestDistinguisher,
#     ANOVADistinguisher,
#     NICVDistinguisher,
#     SNRDistinguisher
# )

from .task import (
    # DistinguisherTask,
    # TemplateTrainTask,
    # LRATask,
    SortBy
)

from .traceset import (
    Container,
    ContainerETS,
    ReaderETS,
    StorerETS,
    ContainerZARR,
    StorerZARR,
    ReaderZARR,
    ReaderTRS,
    ReaderH5
)


# Create a common logger for the whole package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# formatter: how the log will be formatted:
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Write logs on stdout
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

_global_pool = _ThreadPool(psutil.cpu_count())


def set_thread_number(n: int):
    global _global_pool
    if n <= 0 or n > 64:
        raise ValueError(f"Thread count cannot be set to %d, valid range is 1-64" % n)
    _global_pool = _ThreadPool(n)


_DEFAULT_BATCH = [(0, 25000), (1001, 5000),
                  (5001, 2500), (10001, 1000), (50001, 250), (100001, 100)]


def _find_batch_size(nb_samples: int) -> int:
    """ compute default batch size by experiance.
    """
    for i in range(len(_DEFAULT_BATCH)):
        try:
            if nb_samples >= _DEFAULT_BATCH[i][0] and nb_samples < _DEFAULT_BATCH[i + 1][0]:
                return _DEFAULT_BATCH[i][1]
        except IndexError:
            return _DEFAULT_BATCH[-1][1]
