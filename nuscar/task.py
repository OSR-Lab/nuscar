# Copyright(c)  2023. Open Security Research, Inc. - All Rights Reserved

# Open Security Research, Inc. remains the sole owner of this source code copyrights,
# trademark and any applicable intellectual property.

from tqdm.auto import tqdm

import psutil
import inspect
import logging
import numpy as _np
import pandas as _pd
import time
from threading import Thread
import enum
from nuscar.distinguisher import DistinguisherBase
from nuscar.traceset import Container
from nuscar.traceset.base import SUPPORTED_FRAME_TYPES
from IPython.display import display, HTML
import nuscar

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

class SortBy(enum.IntEnum):
    """Enumeration for sort methord for guessing candidate"""

    MAX = 0
    MIN = 1
    MAX_ABS = 2
    MIN_ABS = 3


def _normalize_frame(frame):
    if frame is None:
        return None
    if not isinstance(frame, SUPPORTED_FRAME_TYPES):
        raise IndexError(
            "frame type must be one of {types}".format(
                types=", ".join([str(t) for t in SUPPORTED_FRAME_TYPES]),
            )
        )
    if isinstance(frame, int):
        return [frame]
    return frame


def _frame_x(frame, nb_samples):
    if frame is None:
        return _np.arange(nb_samples)
    if isinstance(frame, slice):
        return _np.arange(nb_samples)[frame]
    if isinstance(frame, range):
        return _np.array(list(frame))
    if isinstance(frame, int):
        return _np.array([frame])
    values = _np.array(frame)
    if values.dtype == bool:
        return _np.arange(nb_samples)[values]
    return values


def _raw_sample_curve(samples, sample_selector, nb_samples):
    samples = _np.asarray(samples)
    if samples.shape[-1] != nb_samples:
        raise ValueError("raw trace sample length must match result sample length")
    rows = samples.reshape(-1, samples.shape[-1])
    if rows.shape[0] != 1:
        raise ValueError("container sample must contain a single raw trace")
    return rows[0, sample_selector]


def _guess_index(guesses, guess):
    matches = _np.nonzero(_np.asarray(guesses) == guess)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def _plot_result_with_traces(
    res,
    guesses,
    dist_name,
    container,
    score_array,
    sortby,
    frame,
    target_idx,
    trace_idx,
    correct_key,
    yzoom,
):
    if target_idx < 0 or target_idx >= res.shape[1]:
        raise ValueError("target_idx out of range")

    if sortby in (SortBy.MAX, SortBy.MAX_ABS):
        best_guess_idx = int(_np.argmax(score_array[:, target_idx]))
    else:
        best_guess_idx = int(_np.argmin(score_array[:, target_idx]))

    sample_selector = frame if frame is not None else slice(None)
    best_guess = int(guesses[best_guess_idx])
    result_curves = [res[best_guess_idx, target_idx, sample_selector]]
    result_names = [f"best {dist_name} 0x{best_guess:02x}"]

    if correct_key is not None:
        correct_guess = int(correct_key[target_idx])
        correct_guess_idx = _guess_index(guesses, correct_guess)
        if correct_guess_idx is not None and correct_guess_idx != best_guess_idx:
            result_curves.append(res[correct_guess_idx, target_idx, sample_selector])
            result_names.append(f"correct {dist_name} 0x{correct_guess:02x}")

    trace_indices = [int(trace_idx)] if _np.isscalar(trace_idx) else list(trace_idx)
    raw_curves = []
    raw_names = []
    for idx in trace_indices:
        raw_curves.append(_raw_sample_curve(container[idx].samples, sample_selector, res.shape[-1]))
        raw_names.append(f"raw trace {idx}")

    x = _frame_x(frame, res.shape[-1])

    return nuscar.plot_multi(
        [_np.asarray(result_curves), _np.asarray(raw_curves)],
        x=x,
        names=[result_names, raw_names],
        title=f"{dist_name} result vs raw traces",
        layout="subplots",
        share_x=True,
        yzoom=yzoom,
    )


def _format_candidates(df, format):
    if format == 'dec':
        return df
    if format == 'hex':
        return df.map(lambda x: f"0x{int(x):02x}")
    raise ValueError("format must be 'dec' or 'hex'")


def _score_array(res, sortby: SortBy = SortBy.MAX_ABS, frame=None):
    frame = _normalize_frame(frame)
    if frame is not None:
        res = res[..., frame]
    if sortby == SortBy.MAX_ABS:
        return _np.max(_np.abs(res), axis=2)
    if sortby == SortBy.MAX:
        return _np.max(res, axis=2)
    if sortby == SortBy.MIN:
        return _np.min(res, axis=2)
    return _np.min(_np.abs(res), axis=2)


class DistinguisherTask:
    """
    This class is top side-channel operation in nuscar.

    It run as follows:

    - get batch of side channel traces from a container
    - using selection function to compute outputs of batchs
    - store result for every distinguisher

    :param container: the container that will be read during the task. Only
        mandatory argument for constructor.
    :param selection_func: the function used to compute intermediate value of a cipher
    :param distinguisher: distinguisher to be registered by the Task.
    :param steps: specify internal steps when the Task will ask its engines to
        compute results.
    :param name: name of the Task.
    :progressbar: True or False, use progressbar or not
    """

    def __init__(
        self,
        container: Container,
        selection_func,
        distinguisher=None,
        steps=None,
        name="Dist Task",
        progressbar=True,
    ):
        self.logger = logging.getLogger(__name__.split('.')[0])
        self.logger.debug("creating task %s", name)
        self.name = name
        self.distinguishers = []
        self.step_result = {}
        self.result = None
        self.container = container
        self._progressbar = progressbar
        self._sel_func = selection_func
        if isinstance(distinguisher, list):
            self.add_distinguishers(distinguisher)
        else:
            self.add_distinguisher(distinguisher)

        self.output_steps = steps
        self._is_sel_with_guess = False

    @property
    def output_steps(self):
        return self._output_steps

    @output_steps.setter
    def output_steps(self, output_steps):
        if isinstance(output_steps, int):
            self._output_steps = list(
                range(output_steps, len(self.container) + 1, output_steps)
            )
        elif hasattr(output_steps, "__iter__"):
            self._output_steps = [i for i in output_steps]
        else:
            self._output_steps = []

        if not len(self.container) in self._output_steps:
            self._output_steps.append(len(self.container))
        self._output_steps.sort()

    def _set_final_result(self):
        self.result = self.step_result[self.output_steps[-1]]
        if isinstance(self.result, list):
            self.result = [
                result if result.flags['C_CONTIGUOUS'] else _np.ascontiguousarray(result)
                for result in self.result
            ]
        elif not self.result.flags['C_CONTIGUOUS']:
            self.result = _np.ascontiguousarray(self.result)

    def add_distinguisher(self, dist: DistinguisherBase):
        """
        Add a distinguisher to the task

        :param distinguisher: distinguisher to be added
        :return: None
        """
        self.distinguishers.append(dist)

    def add_distinguishers(self, dists: list):
        """
        Add a list of distinguishers to the task

        :param distinguishers: list of distinguisher to be added
        :return: None
        """
        for dist in dists:
            self.add_distinguisher(dist)

    def _generate_batch_offsets(self, batch_size):
        """
        From a maximum batch_size
        :param batch_size:
        :return:
        """

        batch_offsets = []
        offset = 0

        while offset < len(self.container):
            if offset + batch_size > len(self.container):
                batch_offsets.append((offset, len(self.container)))

            else:
                batch_offsets.append((offset, offset + batch_size))
            offset += batch_size

        output_steps = list(self._output_steps)[::-1]

        for output_step in output_steps:

            for i, offsets in enumerate(batch_offsets):
                if offsets[0] < output_step < offsets[1]:
                    batch_offsets = (
                        batch_offsets[:i]
                        + [(offsets[0], output_step), (output_step, offsets[1])]
                        + batch_offsets[i + 1:]
                    )

        return batch_offsets

    def _run_sel_func(self, batch) -> _np.ndarray:
        if self._sel_func is None:
            return None
        param_value = []
        for param in self._sel_params:
            param_value.append(getattr(batch, param))
        param_dict = dict(zip(self._sel_params, param_value))
        return self._sel_func(**param_dict)

    def _check_sel_func(self):
        """test selection function.
        """
        self._sel_params = []
        self.nb_sel_target = 1
        if self._sel_func is not None:
            sig = inspect.signature(self._sel_func)
            self.logger.debug(f"selection function signature is %s" % sig)
            for param in sig.parameters.values():
                if param.name == 'guesses':
                    if not param.default is param.empty:  # User function has default value for guesses
                        self.guesses = list(param.default)
                    else:
                        raise ValueError(f"Must provide default value for guesses to perform guessing")
                    self._is_sel_with_guess = True
                    continue
                if param.default is param.empty:
                    if not param.name in self.container[0].metadatas:
                        raise ValueError(
                            f"Trace does not have [%s] attribute" % param.name)
                    self._sel_params.append(param.name)
            n_check = min(2, len(self.container))
            res = self._run_sel_func(self.container[0:n_check])
            if self._is_sel_with_guess:
                assert (len(res.shape) == 3)
                assert (res.shape[0] == n_check and res.shape[1] == len(self.guesses))
                self.nb_sel_target = res.shape[2]
            else:
                assert (len(res.shape) == 2)
                assert (res.shape[0] == n_check)
                self.nb_sel_target = res.shape[1]
        self._nb_samples = self.container[0].samples.shape[-1]
        self.logger.info("check selection func pass")

    def _check_memory(self):
        threshold = 0.9  # we arbitrarily take 90% of the available memory
        available_memory = psutil.virtual_memory().available * threshold

        result_mem = len(self.distinguishers) * \
            len(self.output_steps) * 8 * self.nb_sel_target * \
            2 * self._nb_samples
        if self._is_sel_with_guess:
            result_mem = result_mem*len(self.guesses)
        batch_mem = self._batch_size * 8 * self._nb_samples
        mem_req = result_mem + batch_mem
        if mem_req > available_memory:
            raise MemoryError(f"Insufficient memory, required [%.2fMB], system available [%.2fMB]" %
                              (mem_req/1024.0/1024.0, available_memory/1024.0/1024.0))
        self.logger.info("check memory pass")

    def _do_next_batch(self, next_offsets):
        batch = self.container[next_offsets[0]: next_offsets[1]]
        self._next_batch_samples = batch.samples
        self._next_sel_output = self._run_sel_func(batch)
        self._next_batch = batch

    def run(self, batch_size: int = 0):
        """Run the task over the container and store distinguisher results.

        Args:
            batch_size (int, optional): Number of traces per processing batch. ``0`` selects an automatic batch size.
        """
        for dist in self.distinguishers:
            dist.is_initialized = False
        if self._progressbar:
            pbar = tqdm(total=len(self.container))
        self._check_sel_func()
        if batch_size == 0:
            self._batch_size = nuscar._find_batch_size(self._nb_samples)
        else:
            self._batch_size = batch_size
        self._check_memory()

        self.logger.debug(
            "process with #%d offsets.", self._batch_size)

        batch_offsets = self._generate_batch_offsets(self._batch_size)
        self.logger.debug(
            "task run() will be done in %d batchs" % (len(batch_offsets))
        )
        self.logger.info(
            "task %s: %d traces, %d distinguisher, batch_size=%d, nb_samples=%s"
            % (
                self.name,
                len(self.container),
                len(self.distinguishers),
                self._batch_size,
                self._nb_samples,
            )
        )
        start_time = time.time()
        for i, offsets in enumerate(batch_offsets):
            self.logger.debug(
                "processing batch #%d/%d, with trace offsets: %s."
                % (i + 1, len(batch_offsets), str(offsets))
            )
            if i == 0:
                batch = self.container[offsets[0]: offsets[1]]
                # Calculate intermediate result sel_output
                # _is_sel_with_guess: shape (batch_size, nb_guess, nb_sel_target)
                # else: shape (batch_size, nb_sel_target)
                sample = batch.samples
                sel_output = self._run_sel_func(batch)

            else:
                next_batch_thread.join()
                batch = self._next_batch
                sample = self._next_batch_samples
                sel_output = self._next_sel_output

            if sample.ndim == 1:
                sample = sample[_np.newaxis, :]

            if i+1 < len(batch_offsets):
                next_batch_thread = Thread(
                    target=self._do_next_batch, args=(batch_offsets[i+1],))
                next_batch_thread.start()

            for dist in self.distinguishers:
                if sel_output is None:
                    dist.update(sample, sel_output)
                else:
                    dist.update(sample, sel_output.reshape(
                        len(batch), -1))  # distinguisher input is 2D
            if offsets[1] and offsets[1] in self.output_steps:
                self.logger.debug(
                    "computing results (output step %d)." % offsets[1])
                if len(self.distinguishers) > 1:
                    res_list = []
                    for dist in self.distinguishers:
                        if self._is_sel_with_guess:
                            res_list.append(
                                dist.final().reshape(len(self.guesses), self.nb_sel_target, -1))
                        else:
                            res_list.append(dist.final())
                    self.step_result[offsets[1]] = res_list
                else:
                    if self._is_sel_with_guess:
                        self.step_result[offsets[1]] = self.distinguishers[0].final().reshape(
                            len(self.guesses), self.nb_sel_target, -1)
                    else:
                        self.step_result[offsets[1]
                                         ] = self.distinguishers[0].final()

            if self._progressbar:
                pbar.update(offsets[1] - offsets[0])
        if self._progressbar:
            pbar.close()
        end_time = time.time()
        self.logger.info("Task completed, time taken: %.1f seconds" % (end_time - start_time))
        self._set_final_result()

    def _score(self, dist_idx: int = 0, sortby: SortBy = SortBy.MAX_ABS, frame=None) -> dict:
        if not self._is_sel_with_guess:
            self.logger.warn("No guess values, cannot perform sorting")
            return
        frame = _normalize_frame(frame)
        step_score = {}
        for s, r in self.step_result.items():
            if len(self.distinguishers) > 1:
                res = r[dist_idx]
            else:
                res = r
            step_score[s] = _score_array(res, sortby=sortby, frame=frame)
        return step_score

    def _candidate(self, score: _np.ndarray, sortby: SortBy = SortBy.MAX_ABS, top=None) -> _np.ndarray:
        """calculate candidate from given score and 

        Args:
            score (ndarray): len(guesses) * len(nb_sel_target)
            sortby (SortBy, optional): Defaults to SortBy.MAX_ABS.
            top (_type_, optional): how many candidates to output Defaults to None.
        """
        guess_array = _np.array(self.guesses)
        if top is None:
            if sortby == SortBy.MAX or sortby == SortBy.MAX_ABS:
                return guess_array[_np.argmax(score, axis=0)]
            else:
                return guess_array[_np.argmin(score, axis=0)]
        else:
            if top > len(guess_array):
                top = len(guess_array)
            if sortby == SortBy.MAX or sortby == SortBy.MAX_ABS:
                # Should first take top, then index
                return guess_array[_np.argsort(score, axis=0)[-top:, :][::-1, :]]
            else:
                return guess_array[_np.argsort(score, axis=0)[:top, :]]

    def show_result(self, correct_key: _np.ndarray = None, sortby: SortBy = SortBy.MAX_ABS, frame=None, plot_type="scatter", target_idx=0):
        """Display attack results for guessing distinguishers.

        Args:
            correct_key (ndarray, optional): Correct key values for each target byte.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring and curve display.
            plot_type (str, optional): ``"scatter"`` for score scatter plots or ``"curve"`` for guess curves. Defaults to ``"scatter"``.
            target_idx (int, optional): Target byte/index to display when ``plot_type="curve"``. Defaults to 0.

        Returns:
            ipywidgets.VBox or None: Plotly widget for ``plot_type="curve"``; scatter mode displays directly.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if not self._is_sel_with_guess:
            if len(self.distinguishers) > 1:
                for i, res in enumerate(self.result):
                    display(nuscar.plot(res, title=self.distinguishers[i].name))
            else:
                return nuscar.plot(self.result, title=self.distinguishers[0].name)
        else:
            frame = _normalize_frame(frame)
            for i, d in enumerate(self.distinguishers):
                display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
                res = self.result[i] if len(self.distinguishers) > 1 else self.result
                score_array = _score_array(res, sortby=sortby, frame=frame)
                candidates = self._candidate(
                    score_array, sortby=sortby, top=1)
                # _candidate with top returns 2D array, take first row
                if candidates.ndim == 2:
                    candidates = candidates[0]
                if plot_type == "scatter":
                    nuscar.view._scatter_score(
                        score_array, self.guesses, candidates, correct_key)
                elif plot_type == "curve":
                    if target_idx < 0 or target_idx >= len(candidates):
                        raise ValueError("target_idx out of range")
                    x = _frame_x(frame, res.shape[-1])
                    if frame is not None:
                        res = res[..., frame]
                    return nuscar.view._candidate_curve(
                        res, self.guesses, candidates[target_idx], target_idx, correct_key, x)
                else:
                    raise ValueError("plot_type must be 'scatter' or 'curve'")

    def plot_result_with_traces(
        self,
        correct_key: _np.ndarray = None,
        sortby: SortBy = SortBy.MAX_ABS,
        frame=None,
        target_idx: int = 0,
        trace_idx=0,
        dist_idx: int = 0,
        yzoom: bool = True,
    ):
        """Plot best/correct result curves with raw traces on linked x-axis subplots.

        Args:
            correct_key: Correct key values for each target byte. When provided,
                the correct candidate curve is shown with the best candidate
                curve and raw traces.
            sortby: Score reduction used to rank guesses. Defaults to
                ``SortBy.MAX_ABS``.
            frame: Optional sample/time positions to display.
            target_idx: Target byte/index to display. Defaults to 0.
            trace_idx: Raw trace index, or an iterable of raw trace indices,
                from ``container``. Defaults to 0.
            dist_idx: Distinguisher index when the task has multiple
                distinguishers. Defaults to 0.
            yzoom: Allow y-axis zoom in the returned subplots. Defaults to True.

        Returns:
            ipywidgets.VBox: Widget returned by ``plot_multi``.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if not self._is_sel_with_guess:
            raise RuntimeError("plot_result_with_traces only applies to guessing tasks")

        if dist_idx < 0 or dist_idx >= len(self.distinguishers):
            raise ValueError("dist_idx out of range")

        frame = _normalize_frame(frame)
        res = self.result[dist_idx] if len(self.distinguishers) > 1 else self.result
        score_array = _score_array(res, sortby=sortby, frame=frame)
        return _plot_result_with_traces(
            res,
            self.guesses,
            self.distinguishers[dist_idx].name,
            self.container,
            score_array,
            sortby,
            frame,
            target_idx,
            trace_idx,
            correct_key,
            yzoom,
        )

    def show_step_result(self, correct_key: _np.ndarray = None, sortby: SortBy = SortBy.MAX_ABS, frame=None):
        """Display score evolution across configured output steps.

        Args:
            correct_key (ndarray, optional): Correct key values for each target byte.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if not self._is_sel_with_guess:
            raise RuntimeError("Current display only applies to attack analysis tasks!")
        if correct_key is not None:
            if len(correct_key) != self.nb_sel_target:
                raise ValueError("Correct key length mismatch")
        if len(self.output_steps) == 1:
            raise ValueError("No step info, please use show_result to view results")

        for i, d in enumerate(self.distinguishers):
            display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
            score = self._score(i, frame=frame, sortby=sortby)
            step_score_array = _np.empty((
                len(self.guesses), self.nb_sel_target, len(self.output_steps)))
            for i, s in enumerate(self.output_steps):
                step_score_array[:, :, i] = score[s]
            candidates = self._candidate(
                step_score_array[:, :, -1], sortby=sortby, top=1)
            # _candidate with top returns 2D array, take first row
            if candidates.ndim == 2:
                candidates = candidates[0]
            nuscar.view._plot_step_score(
                step_score_array, self.guesses, self.output_steps, candidates, correct_key)

    def show_candidate(self, top: int = 3, correct_key: _np.ndarray = None, sortby: SortBy = SortBy.MAX_ABS, frame=None, format='dec'):
        """Display top-ranked key candidates as a table.

        Args:
            top (int, optional): Number of candidate ranks to display. Defaults to 3.
            correct_key (ndarray, optional): Correct key values for highlighting.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring.
            format (str, optional): Candidate display format, ``"dec"`` or ``"hex"``. Defaults to ``"dec"``.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if correct_key is not None:
            if len(correct_key) != self.nb_sel_target:
                raise ValueError("Correct key length mismatch")

        for i, d in enumerate(self.distinguishers):
            if not self._is_sel_with_guess:
                display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
                candidates = _np.argsort(-self.result[i])[:top]
            else:
                display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
                res = self.result[i] if len(self.distinguishers) > 1 else self.result
                score_array = _score_array(res, sortby=sortby, frame=frame)
                candidates = self._candidate(
                    score_array, sortby=sortby, top=top)
            index = [f'rank %d' % (i+1) for i in range(len(candidates))]
            if top > len(candidates):
                top = len(candidates)
            candidate_df = _pd.DataFrame(candidates, index=index)
            df = _format_candidates(candidate_df, format)
            if correct_key is not None:
                def highlight_correct(x):
                    styles = _pd.DataFrame('', index=df.index, columns=df.columns)
                    for j in range(len(correct_key)):
                        for i in range(top):
                            if candidate_df.iloc[i, j] == correct_key[j]:
                                styles.iloc[i, j] = 'background-color: #e6ffe6; color:black; font-weight: bold;'
                    return styles
                dfs = df.style.apply(highlight_correct, axis=None)
                dfs.set_table_styles(
                    [{'selector': 'td:hover',
                      'props': [('background-color', 'orange')]}]
                )
                display(dfs)
            else:
                dfs = df.style.set_table_styles(
                    [{'selector': 'td:hover',
                      'props': [('background-color', 'orange')]}]
                )
                display(dfs)

    def get_top_candidate(self, top: int = 1, sortby: SortBy = SortBy.MAX_ABS, frame=None):
        """Return top-ranked candidates for each distinguisher.

        Args:
            top (int, optional): Number of candidate ranks to return. Defaults to 1.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring.

        Returns:
            list: Candidate arrays, one per distinguisher.
        """
        candidates = []
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        for i, d in enumerate(self.distinguishers):
            if not self._is_sel_with_guess:
                candidate = _np.argsort(-self.result[i])[:top]
            else:
                res = self.result[i] if len(self.distinguishers) > 1 else self.result
                score_array = _score_array(res, sortby=sortby, frame=frame)
                candidate = self._candidate(
                    score_array, sortby=sortby, top=top)
            candidates.append(candidate)
        return candidates


class TemplateTrainTask:
    """
    This class is top side-channel operation in nuscar to train a template model.

    It run as follows:

    - get batch of side channel traces from a container
    - using selection function to compute outputs of batchs
    - store template model

    :param container: the container that will be read during the task. Only
        mandatory argument for constructor.
    :param selection_func: the function used to compute intermediate value of a cipher
    :param distinguisher: template distinguisher to be registered by the Task.
    :param name: name of the Task.
    :progressbar: True or False, use progressbar or not
    """

    def __init__(
        self,
        container: Container,
        selection_func,
        distinguisher=None,
        name="Template Task",
        progressbar=True,
    ):
        self.logger = logging.getLogger(__name__.split('.')[0])
        self.logger.debug("creating task %s", name)
        self.name = name
        self.distinguishers = []
        self.model = None
        self.container = container
        self._progressbar = progressbar
        self._sel_func = selection_func
        if isinstance(distinguisher, list):
            self.add_distinguishers(distinguisher)
        else:
            self.add_distinguisher(distinguisher)

    def add_distinguisher(self, dist: DistinguisherBase):
        """
        Add a distinguisher to the task

        :param distinguisher: distinguisher to be added
        :return: None
        """
        self.distinguishers.append(dist)

    def add_distinguishers(self, dists: list):
        """
        Add a list of distinguishers to the task

        :param distinguishers: list of distinguisher to be added
        :return: None
        """
        for dist in dists:
            self.add_distinguisher(dist)

    def _generate_batch_offsets(self, batch_size):
        """
        From a maximum batch_size, this function computes the offsets of the batchs that will be used by the task.run() method.

        :param batch_size:
        :return:
        """

        batch_offsets = []
        offset = 0

        while offset < len(self.container):
            if offset + batch_size > len(self.container):
                batch_offsets.append((offset, len(self.container)))

            else:
                batch_offsets.append((offset, offset + batch_size))
            offset += batch_size
        return batch_offsets

    def _run_sel_func(self, batch) -> _np.ndarray:
        param_value = []
        for param in self._sel_params:
            param_value.append(getattr(batch, param))
        param_dict = dict(zip(self._sel_params, param_value))
        return self._sel_func(**param_dict)

    def _check_sel_func(self):
        """test selection function.
        """
        self._sel_params = []
        sig = inspect.signature(self._sel_func)
        self.logger.debug(f"selection function signature is %s" % sig)
        for param in sig.parameters.values():
            if param.default is param.empty:
                if not param.name in self.container[0].metadatas:
                    raise ValueError(
                        f"Trace does not have [%s] attribute" % param.name)
                self._sel_params.append(param.name)
        n_check = min(2, len(self.container))
        res = self._run_sel_func(self.container[0:n_check])
        assert (len(res.shape) == 2)
        assert (res.shape[0] == n_check)
        self.nb_sel_target = res.shape[1]
        self._nb_samples = self.container[0].samples.shape[-1]
        self.logger.info("check selection func pass")

    def _check_memory(self):
        threshold = 0.9  # we arbitrarily take 90% of the available memory
        available_memory = psutil.virtual_memory().available * threshold

        result_mem = len(self.distinguishers) * \
            8 * self._nb_samples * \
            2 * self._nb_samples
        batch_mem = self._batch_size * 8 * self._nb_samples
        mem_req = result_mem + batch_mem
        if mem_req > available_memory:
            raise MemoryError(f"Insufficient memory, required [%.2fMB], system available [%.2fMB]" %
                              (mem_req/1024.0/1024.0, available_memory/1024.0/1024.0))
        self.logger.info("check memory pass")

    def _do_next_batch(self, next_offsets):
        batch = self.container[next_offsets[0]: next_offsets[1]]
        self._next_batch_samples = batch.samples
        self._next_sel_output = self._run_sel_func(batch)
        self._next_batch = batch

    def run(self, batch_size: int = 0, epoch=1):
        """Train template distinguishers over one or more epochs.

        Args:
            batch_size (int, optional): Number of traces per processing batch. ``0`` selects an automatic batch size.
            epoch (int, optional): Number of passes over the container. Defaults to 1.
        """
        for dist in self.distinguishers:
            dist.is_initialized = False
        if self._progressbar:
            pbar = tqdm(total=len(self.container)*epoch)
        self._check_sel_func()
        if batch_size == 0:
            self._batch_size = nuscar._find_batch_size(self._nb_samples)
        else:
            self._batch_size = batch_size
        self._check_memory()

        self.logger.debug(
            "process with #%d offsets.", self._batch_size)

        batch_offsets = self._generate_batch_offsets(self._batch_size)
        self.logger.debug(
            "task run() will be done in %d batchs" % (len(batch_offsets))
        )
        self.logger.info(
            "task %s: %d traces, %d distinguisher, batch_size=%d, nb_samples=%s"
            % (
                self.name,
                len(self.container),
                len(self.distinguishers),
                self._batch_size,
                self._nb_samples,
            )
        )
        start_time = time.time()
        for e in range(epoch):
            for i, offsets in enumerate(batch_offsets):
                self.logger.debug(
                    "processing batch #%d/%d, with trace offsets: %s."
                    % (i + 1, len(batch_offsets), str(offsets))
                )
                if i == 0:
                    batch = self.container[offsets[0]: offsets[1]]
                    # Calculate intermediate result sel_output
                    # _is_sel_with_guess: shape (batch_size, nb_guess, nb_sel_target)
                    # else: shape (batch_size, nb_sel_target)
                    sample = batch.samples
                    sel_output = self._run_sel_func(batch)

                else:
                    next_batch_thread.join()
                    batch = self._next_batch
                    sample = self._next_batch_samples
                    sel_output = self._next_sel_output

                if i+1 < len(batch_offsets):
                    next_batch_thread = Thread(
                        target=self._do_next_batch, args=(batch_offsets[i+1],))
                    next_batch_thread.start()

                for dist in self.distinguishers:
                    dist.update(sample, sel_output.reshape(
                        len(batch), -1))  # distinguisher input is 2D
                if self._progressbar:
                    pbar.update(offsets[1] - offsets[0])

        for dist in self.distinguishers:
            dist.final()
        if self._progressbar:
            pbar.close()
        end_time = time.time()
        self.logger.info("Task completed, time taken: %.1f seconds" % (end_time - start_time))


class LRATask:
    """
    This class used to do LRA as it has different logic with other distinguier.

    It run as follows:

    - get batch of side channel traces from a container
    - using selection function to compute outputs of batchs
    - store result for lra distinguisher

    :param container: the container that will be read during the task. Only
        mandatory argument for constructor.
    :param partition_func: the function used to do partition for LRA
    :param distinguisher: lra distinguisher to be registered by the Task.
    :param steps: specify internal steps when the Task will ask its engines to
        compute results.
    :param name: name of the Task.
    :progressbar: True or False, use progressbar or not
    """

    def __init__(
        self,
        container: Container,
        partition_func,
        distinguisher=None,
        steps=None,
        name="Dist Task",
        progressbar=True,
    ):
        self.logger = logging.getLogger(__name__.split('.')[0])
        self.logger.debug("creating task %s", name)
        self.name = name
        self.distinguisher = distinguisher
        self.step_result = {}
        self.result = None
        self.container = container
        self._progressbar = progressbar
        self._sel_func = partition_func
        self.output_steps = steps

    @property
    def output_steps(self):
        return self._output_steps

    @output_steps.setter
    def output_steps(self, output_steps):
        if isinstance(output_steps, int):
            self._output_steps = list(
                range(output_steps, len(self.container) + 1, output_steps)
            )
        elif hasattr(output_steps, "__iter__"):
            self._output_steps = [i for i in output_steps]
        else:
            self._output_steps = []

        if not len(self.container) in self._output_steps:
            self._output_steps.append(len(self.container))
        self._output_steps.sort()


    def _generate_batch_offsets(self, batch_size):
        """
        From a maximum batch_size
        :param batch_size:
        :return:
        """

        batch_offsets = []
        offset = 0

        while offset < len(self.container):
            if offset + batch_size > len(self.container):
                batch_offsets.append((offset, len(self.container)))

            else:
                batch_offsets.append((offset, offset + batch_size))
            offset += batch_size

        output_steps = list(self._output_steps)[::-1]

        for output_step in output_steps:

            for i, offsets in enumerate(batch_offsets):
                if offsets[0] < output_step < offsets[1]:
                    batch_offsets = (
                        batch_offsets[:i]
                        + [(offsets[0], output_step), (output_step, offsets[1])]
                        + batch_offsets[i + 1:]
                    )

        return batch_offsets

    def _run_sel_func(self, batch) -> _np.ndarray:
        param_value = []
        for param in self._sel_params:
            param_value.append(getattr(batch, param))
        param_dict = dict(zip(self._sel_params, param_value))
        return self._sel_func(**param_dict)

    def _check_sel_func(self):
        """test selection function.
        """
        self._sel_params = []
        sig = inspect.signature(self._sel_func)
        self.logger.debug(f"selection function signature is %s" % sig)
        self.guesses = self.distinguisher.guesses # lra has guesses attribute

        for param in sig.parameters.values():
            if param.default is param.empty:
                if not param.name in self.container[0].metadatas:
                    raise ValueError(
                        f"Trace does not have [%s] attribute" % param.name)
                self._sel_params.append(param.name)
        n_check = min(2, len(self.container))
        res = self._run_sel_func(self.container[0:n_check])

        assert (len(res.shape) == 2)
        assert (res.shape[0] == n_check)
        self.nb_sel_target = res.shape[1]
        self._nb_samples = self.container[0].samples.shape[-1]
        self.logger.info("check selection func pass")

    def _check_memory(self):
        threshold = 0.9  # we arbitrarily take 90% of the available memory
        available_memory = psutil.virtual_memory().available * threshold

        result_mem = len(self.output_steps) * 8 * self.nb_sel_target * \
            2 * self._nb_samples
        
        batch_mem = self._batch_size * 8 * self._nb_samples
        mem_req = result_mem + batch_mem
        if mem_req > available_memory:
            raise MemoryError(f"Insufficient memory, required [%.2fMB], system available [%.2fMB]" %
                              (mem_req/1024.0/1024.0, available_memory/1024.0/1024.0))
        self.logger.info("check memory pass")

    def _do_next_batch(self, next_offsets):
        batch = self.container[next_offsets[0]: next_offsets[1]]
        self._next_batch_samples = batch.samples
        self._next_sel_output = self._run_sel_func(batch)
        self._next_batch = batch

    def run(self, batch_size: int = 0):
        """Run the LRA task over the container and store step results.

        Args:
            batch_size (int, optional): Number of traces per processing batch. ``0`` selects an automatic batch size.
        """
        self.distinguisher.is_initialized = False
        if self._progressbar:
            pbar = tqdm(total=len(self.container))
        self._check_sel_func()
        if batch_size == 0:
            self._batch_size = nuscar._find_batch_size(self._nb_samples)
        else:
            self._batch_size = batch_size
        self._check_memory()

        self.logger.debug(
            "process with #%d offsets.", self._batch_size)

        batch_offsets = self._generate_batch_offsets(self._batch_size)
        self.logger.debug(
            "task run() will be done in %d batchs" % (len(batch_offsets))
        )
        self.logger.info(
            "task %s: %d traces, LRA distinguisher, batch_size=%d, nb_samples=%s"
            % (
                self.name,
                len(self.container),
                self._batch_size,
                self._nb_samples,
            )
        )
        start_time = time.time()
        for i, offsets in enumerate(batch_offsets):
            self.logger.debug(
                "processing batch #%d/%d, with trace offsets: %s."
                % (i + 1, len(batch_offsets), str(offsets))
            )
            if i == 0:
                batch = self.container[offsets[0]: offsets[1]]
                # Calculate intermediate result sel_output
                # _is_sel_with_guess: shape (batch_size, nb_guess, nb_sel_target)
                # else: shape (batch_size, nb_sel_target)
                sample = batch.samples
                sel_output = self._run_sel_func(batch)

            else:
                next_batch_thread.join()
                batch = self._next_batch
                sample = self._next_batch_samples
                sel_output = self._next_sel_output

            if sample.ndim == 1:
                sample = sample[_np.newaxis, :]

            if i+1 < len(batch_offsets):
                next_batch_thread = Thread(
                    target=self._do_next_batch, args=(batch_offsets[i+1],))
                next_batch_thread.start()

            
            self.distinguisher.update(sample, sel_output.reshape(
                    len(batch), -1))  # distinguisher input is 2D
            if offsets[1] and offsets[1] in self.output_steps:
                self.logger.debug(
                    "computing results (output step %d)." % offsets[1])
                if self.guesses:
                    self.step_result[offsets[1]] = self.distinguisher.final().reshape(
                        len(self.guesses), self.nb_sel_target, -1)
                else:
                    self.step_result[offsets[1]] = self.distinguisher.final()

            if self._progressbar:
                pbar.update(offsets[1] - offsets[0])
        if self._progressbar:
            pbar.close()
        end_time = time.time()
        self.logger.info("Task completed, time taken: %.1f seconds" % (end_time - start_time))
        self.result = self.step_result[self.output_steps[-1]]
        if not self.result.flags['C_CONTIGUOUS']:
            self.result = _np.ascontiguousarray(self.result)

    def _score(self, sortby: SortBy = SortBy.MAX_ABS, frame=None) -> dict:
        """calculate score of self.step_result.

        Args:
            sortby (SortBy, optional): Defaults to SortBy.MAX_ABS.
            frame (list, optional): Calculate score by slice of result. Defaults to None, means score is calculate by all result.

        Returns:
            dict: 2D array, len(guesses) * len(nb_sel_target)
        """
        if self.guesses is None:
            self.logger.warn("No guess values, cannot perform sorting")
            return
        frame = _normalize_frame(frame)
        step_score = {}
        for s, r in self.step_result.items():
            step_score[s] = _score_array(r, sortby=sortby, frame=frame)
        return step_score

    def _candidate(self, score: _np.ndarray, sortby: SortBy = SortBy.MAX_ABS, top=None) -> _np.ndarray:
        """calculate candidate from given score and 

        Args:
            score (ndarray): len(guesses) * len(nb_sel_target)
            sortby (SortBy, optional): Defaults to SortBy.MAX_ABS.
            top (_type_, optional): how many candidates to output Defaults to None.
        """
        guess_array = _np.array(self.guesses)
        if top is None:
            if sortby == SortBy.MAX or sortby == SortBy.MAX_ABS:
                return guess_array[_np.argmax(score, axis=0)]
            else:
                return guess_array[_np.argmin(score, axis=0)]
        else:
            if top > len(guess_array):
                top = len(guess_array)
            if sortby == SortBy.MAX or sortby == SortBy.MAX_ABS:
                return guess_array[_np.argsort(score, axis=0)[-top:, :][::-1, :]]
            else:
                return guess_array[_np.argsort(score, axis=0)[:top, :]]

    def show_result(self, correct_key: _np.ndarray=None, sortby: SortBy=SortBy.MAX_ABS, frame:list=None, plot_type="scatter", target_idx=0):
        """Display LRA attack results.

        Args:
            correct_key (ndarray, optional): Correct key values for each target byte.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring and curve display.
            plot_type (str, optional): ``"scatter"`` for score scatter plots or ``"curve"`` for guess curves. Defaults to ``"scatter"``.
            target_idx (int, optional): Target byte/index to display when ``plot_type="curve"``. Defaults to 0.

        Returns:
            ipywidgets.VBox or None: Plotly widget for ``plot_type="curve"``; scatter mode displays directly.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if self.guesses is None:
            return nuscar.plot(self.result, title=self.distinguisher.name)
        else:
            frame = _normalize_frame(frame)
            d = self.distinguisher
            display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
            score_array = _score_array(self.result, sortby=sortby, frame=frame)
            candidates = self._candidate(
                score_array, sortby=sortby, top=1)
            # _candidate with top returns 2D array, take first row
            if candidates.ndim == 2:
                candidates = candidates[0]
            if plot_type == "scatter":
                nuscar.view._scatter_score(
                    score_array, self.guesses, candidates, correct_key)
            elif plot_type == "curve":
                if target_idx < 0 or target_idx >= len(candidates):
                    raise ValueError("target_idx out of range")
                res = self.result
                x = _frame_x(frame, res.shape[-1])
                if frame is not None:
                    res = res[..., frame]
                return nuscar.view._candidate_curve(
                    res, self.guesses, candidates[target_idx], target_idx, correct_key, x)
            else:
                raise ValueError("plot_type must be 'scatter' or 'curve'")

    def plot_result_with_traces(
        self,
        correct_key: _np.ndarray = None,
        sortby: SortBy = SortBy.MAX_ABS,
        frame=None,
        target_idx: int = 0,
        trace_idx=0,
        yzoom: bool = True,
    ):
        """Plot best/correct LRA result curves with raw traces on linked x-axis subplots.

        Args:
            correct_key: Correct key values for each target byte. When provided,
                the correct candidate curve is shown with the best candidate
                curve and raw traces.
            sortby: Score reduction used to rank guesses. Defaults to
                ``SortBy.MAX_ABS``.
            frame: Optional sample/time positions to display.
            target_idx: Target byte/index to display. Defaults to 0.
            trace_idx: Raw trace index, or an iterable of raw trace indices,
                from ``container``. Defaults to 0.
            yzoom: Allow y-axis zoom in the returned subplots. Defaults to True.

        Returns:
            ipywidgets.VBox: Widget returned by ``plot_multi``.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if self.guesses is None:
            raise RuntimeError("plot_result_with_traces only applies to guessing tasks")

        frame = _normalize_frame(frame)
        score_array = _score_array(self.result, sortby=sortby, frame=frame)
        return _plot_result_with_traces(
            self.result,
            self.guesses,
            self.distinguisher.name,
            self.container,
            score_array,
            sortby,
            frame,
            target_idx,
            trace_idx,
            correct_key,
            yzoom,
        )

    def show_step_result(self, correct_key:_np.ndarray=None, sortby: SortBy = SortBy.MAX_ABS, frame:list=None):
        """plot result of all distinguishers for all steps.

        Args:
            correct_key (ndarray, optional): correct key, used to compare. Defaults to None.
            sortby (SortBy, optional): Defaults to SortBy.MAX_ABS.
            frame (list, optional): Calculate distiguisher score by frame of result point. Defaults to None, means score is calculate by all result points.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if self.guesses is None:
            raise RuntimeError("Current display only applies to attack analysis tasks!")
        if correct_key is not None:
            if len(correct_key) != self.nb_sel_target:
                raise ValueError("Correct key length mismatch")
        if len(self.output_steps) == 1:
            raise ValueError("No step info, please use show_result to view results")
        d = self.distinguisher
        display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
        score = self._score(frame=frame, sortby=sortby)
        step_score_array = _np.empty((
            len(self.guesses), self.nb_sel_target, len(self.output_steps)))
        for i, s in enumerate(self.output_steps):
            step_score_array[:, :, i] = score[s]
        candidates = self._candidate(
            step_score_array[:, :, -1], sortby=sortby, top=1)
        # _candidate with top returns 2D array, take first row
        if candidates.ndim == 2:
            candidates = candidates[0]
        nuscar.view._plot_step_score(
            step_score_array, self.guesses, self.output_steps, candidates, correct_key)

    def show_candidate(self, top: int = 3, correct_key: _np.ndarray = None, sortby: SortBy = SortBy.MAX_ABS, frame:list=None, format='dec'):
        """Display top-ranked LRA key candidates as a table.

        Args:
            top (int, optional): Number of candidate ranks to display. Defaults to 3.
            correct_key (ndarray, optional): Correct key values for highlighting.
            sortby (SortBy, optional): Score reduction used to rank guesses. Defaults to SortBy.MAX_ABS.
            frame (slice, int, list, ndarray, range, optional): Sample/time positions used for scoring.
            format (str, optional): Candidate display format, ``"dec"`` or ``"hex"``. Defaults to ``"dec"``.
        """
        if self.result is None:
            raise RuntimeError("Task must be executed before displaying results!")
        if self.guesses is None:
            raise RuntimeError("Current display only applies to attack analysis tasks!")
        if correct_key is not None:
            if len(correct_key) != self.nb_sel_target:
                raise ValueError("Correct key length mismatch")
        d = self.distinguisher
        display(HTML(f'<h3>Distinguisher: %s</h3>' % d.name))
        score_array = _score_array(self.result, sortby=sortby, frame=frame)
        candidates = self._candidate(
            score_array, sortby=sortby, top=top)
        index = [f'rank %d' % (i+1) for i in range(len(candidates))]
        if top > len(candidates):
            top = len(candidates)
        candidate_df = _pd.DataFrame(candidates, index=index)
        df = _format_candidates(candidate_df, format)
        if correct_key is not None:
            def highlight_correct(x):
                styles = _pd.DataFrame('', index=df.index, columns=df.columns)
                for j in range(len(correct_key)):
                    for i in range(top):
                        if candidate_df.iloc[i, j] == correct_key[j]:
                            styles.iloc[i, j] = 'background-color: #e6ffe6; color:black; font-weight: bold;'
                return styles
            dfs = df.style.apply(highlight_correct, axis=None)
            dfs.set_table_styles(
                [{'selector': 'td:hover',
                    'props': [('background-color', 'orange')]}]
            )
            display(dfs)
        else:
            dfs = df.style.set_table_styles(
                [{'selector': 'td:hover',
                    'props': [('background-color', 'orange')]}]
            )
            display(dfs)