# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from tqdm.auto import tqdm
from nuscar.traceset import Storer, StorerMemory, ContainerMemory
import logging
import time as _time
import numpy as _np
import plotly.express as _px
from IPython.display import display
import plotly.graph_objects as _go
from plotly_resampler import FigureWidgetResampler
import threading
import ipywidgets as _widgets


def delay(delay):
    """ Function to provide accurate time delay in nanosecond
    """
    _ = _time.perf_counter_ns() + delay
    while _time.perf_counter_ns() < _:
        pass


class Scope:
    _scope = None

    def __init__(self):
        if Scope._scope is not None:
            try:
                Scope._scope.close()
            except:
                pass
        Scope._scope = self

    @property
    def sequence_num(self) -> int:
        pass

    def arm(self) -> bool:
        pass

    def acquire_samples(self) -> list | None:
        pass

    def close(self):
        pass


class Communicator:
    instance = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if Communicator.instance is not None:
            self.logger.warning("Automatically closing already instantiated Communicator")
            try:
                Communicator.instance.close()
            except:
                pass
        Communicator.instance = self

    def get_metadata(self, meta_index=0):
        pass

    def close(self):
        pass


class _OutputWidgetHandler(logging.Handler):
    def __init__(self, out, *args, **kwargs):
        self.out = out
        super(_OutputWidgetHandler, self).__init__(*args, **kwargs)

    def emit(self, record):
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = self.out.outputs + (new_output,)


class CollectTask:
    def __init__(self, scope: Scope, communicator: Communicator, retry=5, predelay: int = 0, postdelay: int = 0,
                 viewfreq: int = 10):
        """CollectTask is used to collect traceset.

        Args:
            scope (Scope): oscilloscope instance, should be initialized
            communicator (Communicator): communicator used to talk with TOE device.
            retry (int): retry time when error occurred.
            predelay (int): delay between arm() of scope and communicate with toe, in nanoseconds.
            postdelay (int): communicate with toe and read samples from scope, in nanoseconds.
            viewfreq (int): view trace
        """
        self._scope = scope
        self._communicator = communicator
        self._has_run = False
        self._retry = retry
        self._pre_delay = predelay
        self._post_delay = postdelay
        self._stop = False  # button control stop
        self._viewfreq = viewfreq
        self.logger = logging.getLogger(__name__)

    def _get_metas(self, n):
        meta = dict()
        trs_idx = self.trs_idx
        for i in range(n):
            m = self._communicator.get_metadata(trs_idx)
            if m is None:
                self.logger.warning("Communicator returned empty value!")
                return None
            if i == 0:
                for k in m:
                    if type(m[k]) == bytes:
                        t = _np.frombuffer(m[k], dtype=_np.uint8)
                    elif type(m[k]) == _np.ndarray:
                        t = m[k]
                    else:
                        raise TypeError("meta data type error")
                    meta[k] = t
            else:
                for k in m:
                    if type(m[k]) == bytes:
                        t = _np.frombuffer(m[k], dtype=_np.uint8)
                    elif type(m[k]) == _np.ndarray:
                        t = m[k]
                    else:
                        raise TypeError("meta data type error")
                    meta[k] = _np.vstack((meta[k], t))
            trs_idx = trs_idx + 1

        return meta

    def _init_view(self, samples: list):
        fig_widgets = []
        color_list = _px.colors.qualitative.Plotly

        for i, s in enumerate(samples):
            fig = FigureWidgetResampler(_go.Figure())
            s = s.reshape(-1, s.shape[-1])
            x = _np.array(range(0, s.shape[-1]))
            fig.add_trace(_go.Scattergl(opacity=0.8,
                                        line_color=color_list[i % len(color_list)]),
                          hf_x=x, hf_y=s[0])
            fig.update_layout(
                plot_bgcolor="white",
                title={"text": f"Trace 0 @ channel {i}"},
                hovermode="x unified",
                height=400
            )

            fig.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                gridcolor='lightgrey',
            )
            fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                gridcolor='lightgrey'
            )
            fig.update_traces(
                hoverinfo="name+x+y",
            )
            fig_widgets.append(fig)
        return fig_widgets

    def _update_view(self, fig_widgets: list, samples: list, idx: int):
        for i, (w, s) in enumerate(zip(fig_widgets, samples)):
            s = s.reshape(-1, s.shape[-1])
            if w.hf_data:
                w.hf_data[-1]['y'] = s[0]
            else:
                w.data[0].y = s[0]
            w.update_layout(
                title={"text": f"Trace {idx} @ channel {i}"},
                height=400
            )
            w.reload_data()

    def _run_with_storer(self, seq: int, pbar, storer: Storer, view, out: _widgets.Output = None, stop_btn=None):
        nb_trs_per_seq = self._scope.sequence_num
        resend = 0
        seq_idx = 0
        fig_widgets = None
        self._stop = False  # button control stop
        out_handler = None
        if out is not None:
            out_handler = _OutputWidgetHandler(out)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            out_handler.setFormatter(formatter)
            self.logger.addHandler(out_handler)

        while seq_idx < seq:
            if not self._scope.arm():
                resend = resend + 1
                self.logger.warning("Oscilloscope arm failed, retrying automatically...")
                if resend > self._retry:
                    pbar.close()
                    raise RuntimeError("Maximum retry attempts exceeded!")
                continue
            delay(self._pre_delay)
            meta = self._get_metas(nb_trs_per_seq)
            delay(self._post_delay)
            samples = self._scope.acquire_samples()  # samples is list of sample
            fail = False
            if samples is None:
                fail = True
            else:
                for s in samples:
                    if s is None:
                        fail = True
                        break
            if fail:
                resend = resend + 1
                self.logger.warning("Oscilloscope failed to read waveform, please check trigger status, retrying automatically...")
                if resend > self._retry:
                    pbar.close()
                    storer.close()
                    raise RuntimeError("Maximum retry attempts exceeded!")
                _time.sleep(1)
                continue
            if view:
                if fig_widgets is None:
                    fig_widgets = self._init_view(samples)
                    for w in fig_widgets:
                        display(w)
                else:
                    if seq_idx % self._viewfreq == 0:
                        self._update_view(fig_widgets, samples, seq_idx * nb_trs_per_seq)

            for s in samples:
                storer.update(samples=s, **meta)
            seq_idx = seq_idx + 1
            resend = 0
            self.trs_idx = self.trs_idx + nb_trs_per_seq
            pbar.update()
            if self._stop:
                self.logger.info(f"Collection aborted, collected {seq_idx} sequence(s)")
                break
        if out is not None:
            self.logger.removeHandler(out_handler)
        storer.close()
        pbar.close()
        if stop_btn is not None:
            stop_btn.close()

    def check(self, seq: int = 5, view=True):
        """check collection with "seq" number of sequence
        """
        self.trs_idx = 0
        tmp_store = StorerMemory()
        pbar = tqdm(total=seq, desc="Collection progress: ", unit=" Sequence(s)",
                    bar_format='{n} {desc} {percentage:3.0f}% {n_fmt}/{total_fmt} Elapsed: {'
                               'elapsed} Remaining: {remaining} Speed: {rate_fmt}{postfix}')
        self._run_with_storer(seq, pbar, tmp_store, view)
        return ContainerMemory(tmp_store)

    def run(self, storer: Storer, seq, view=True):
        '''run collection with "seq" number of sequence
        Args:
            storer (Storer): storer used to store traceset. After run(), storer is closed by task. 
                run() can only be called once. 
        '''
        self.trs_idx = 0
        pbar = tqdm(total=seq, desc="Collection progress: ", unit=" Sequence(s)",
                    bar_format='{n} {desc} {percentage:3.0f}% {n_fmt}/{total_fmt} Elapsed: {'
                               'elapsed} Remaining: {remaining} Speed: {rate_fmt}{postfix}')
        self._stop = False
        stop_btn = _widgets.Button(
            description='Stop Collection',
            disabled=False,
            button_style='danger',
            icon='stop'
        )

        def _stop(b):
            self.logger.info("Preparing to stop collection...")
            self._stop = True
            stop_btn.disabled = True

        stop_btn.on_click(_stop)
        display(stop_btn)
        out = _widgets.Output()
        display(out)
        thread = threading.Thread(target=self._run_with_storer, args=(seq, pbar, storer, view, out, stop_btn))
        thread.start()
