"""Interactive plotting helpers for waveform and side-channel analysis data.

This module provides notebook-oriented Plotly widgets for traces, score arrays,
scan heatmaps and spectrograms of large waveform data. Most
functions accept NumPy arrays and return a Plotly ``FigureWidget`` or an
``ipywidgets`` container whose underlying figure is available as ``.fig``.
"""

import plotly.graph_objects as _go
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler
import ipyevents as events
from scipy import signal
import plotly.express as _px
from typing import Union
import math
from IPython.display import display
import numpy as _np
import ipywidgets as _widgets

# ============= Color Scales =============

SINGLE_GREEN = [
    [0.0, "rgb(0, 0, 0)"],
    [0.1, "rgb(0, 25, 0)"],
    [0.3, "rgb(0, 80, 0)"],
    [0.5, "rgb(0, 140, 0)"],
    [0.7, "rgb(0, 200, 0)"],
    [1.0, "rgb(0, 255, 0)"],
]

GRADIENT_BLUE_GREEN = [
    [0.0, "rgb(0, 0, 0)"],
    [0.1, "rgb(0, 0, 50)"],
    [0.3, "rgb(0, 0, 150)"],
    [0.5, "rgb(0, 100, 100)"],
    [0.7, "rgb(0, 180, 50)"],
    [1.0, "rgb(0, 255, 0)"],
]

RAINBOW = [
    [0.0, "rgb(0, 0, 0)"],
    [0.1, "rgb(0, 0, 255)"],
    [0.25, "rgb(0, 255, 255)"],
    [0.4, "rgb(0, 255, 0)"],
    [0.55, "rgb(255, 255, 0)"],
    [0.7, "rgb(255, 128, 0)"],
    [0.85, "rgb(255, 0, 0)"],
    [1.0, "rgb(255, 0, 255)"],
]

COLORSCALES = {
    "green": SINGLE_GREEN,
    "gradient": GRADIENT_BLUE_GREEN,
    "rainbow": RAINBOW,
}



def plot(y: _np.ndarray, x: _np.ndarray = None, name: list = None, title: str = None, webgl=True, resample="auto", shift=False):
    """Plot one array as one or more traces in a single interactive figure.

    A 1-D input becomes one trace. A 2-D input is reshaped to
    ``(n_traces, n_samples)`` and each row becomes one trace. For large inputs,
    ``resample="auto"`` enables ``plotly-resampler`` to keep notebook rendering
    responsive.

    Keyboard controls are attached to the returned widget: arrow keys zoom or
    pan the x-axis, ``Alt`` + arrows control the y-axis, ``Shift`` makes smaller
    steps, ``Home`` resets the view, and ``F`` toggles figure height.

    Args:
        y: Samples to plot. The last dimension is the sample axis.
        x: Optional shared 1-D x-axis. Defaults to sample indices.
        name: Optional trace names. Length must match the number of plotted
            rows.
        title: Figure title.
        webgl: Use Plotly ``Scattergl`` traces for faster rendering. Defaults
            to True.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it when
            ``y.size > 100000``. Defaults to ``"auto"``.
        shift: Add controls for shifting one selected trace along the x-axis.
            Defaults to False.

    Returns:
        An ``ipywidgets.VBox`` containing the Plotly figure and controls. The
        underlying figure is available as ``container.fig``.
    """
    if resample == "auto":
        if y.size > 100000:
            resample = True
        else:
            resample = False
    
    y = y.reshape(-1, y.shape[-1])  # change to 2D
    nsamples = y.shape[-1]
    ntraces = y.shape[0]
    if x is None:
        x = _np.array(range(0, nsamples))
    color_list = _px.colors.qualitative.Plotly
    rangeslider = True
    if resample:
        fig = FigureWidgetResampler(_go.Figure())
    else:
        fig = _go.FigureWidget()
    if name is not None:
        assert len(name) == len(y)
    else:
        name = [str(i) for i in range(ntraces)]
    for i, s in enumerate(y):
        if resample:
            s_contiguous = s if s.flags['C_CONTIGUOUS'] else _np.ascontiguousarray(s)
            fig.add_trace(
                _go.Scattergl(name=name[i], opacity=0.8,
                              line_color=color_list[i % len(color_list)]),
                hf_x=x, hf_y=s_contiguous
            )
        else:
            if webgl:
                fig.add_trace(
                    _go.Scattergl(x=x, y=s, name=name[i], opacity=0.8,
                                line_color=color_list[i % len(color_list)])
                )
            else:
                fig.add_trace(
                    _go.Scatter(x=x, y=s, name=name[i], opacity=0.8,
                                line_color=color_list[i % len(color_list)])
                )
    fig.update_layout(
        title=dict(text=title),
        plot_bgcolor="white",
        hovermode="x unified",
        height=400
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        rangeslider=dict(
            visible=rangeslider
        )
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        fixedrange=True
    )
    fig.update_traces(
        hoverinfo="name+x+y",
    )
    def interact_plot(event):
        if event['event'] == 'keydown':
            key = event.get('key', '')
            alt_key = event.get('altKey', False)
            shift_key = event.get('shiftKey', False)

            if key == 'F':
                if fig.layout.height < 800:
                    fig.update_layout(height=800)
                else:
                    fig.update_layout(height=400)
            elif key == 'y':
                fig.update_yaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', fixedrange=False
                )
                fig.update_xaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', fixedrange=True,
                    rangeslider=dict(visible=rangeslider)
                )
            elif key == 'Home':
                fig.update_xaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', autorange=True,
                    rangeslider=dict(visible=rangeslider)
                )
                fig.update_yaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', fixedrange=True, autorange=True
                )
            elif alt_key:
                move_factor = 0.1 if shift_key else 0.5
                zoom_factor = 1.2 if shift_key else 2.0
                y_range = fig.layout.yaxis.range
                if y_range is not None:
                    y_min, y_max = y_range
                    y_center = (y_min + y_max) / 2
                    y_half = (y_max - y_min) / 2
                    if key == 'ArrowUp':
                        y_half /= zoom_factor
                    elif key == 'ArrowDown':
                        y_half *= zoom_factor
                    elif key == 'ArrowRight':
                        shift_val = y_half * 2 * move_factor
                        y_center += shift_val
                    elif key == 'ArrowLeft':
                        shift_val = y_half * 2 * move_factor
                        y_center -= shift_val
                    fig.update_yaxes(
                        range=[y_center - y_half, y_center + y_half],
                        fixedrange=False
                    )
            else:
                move_factor = 0.1 if shift_key else 0.5
                zoom_factor = 1.2 if shift_key else 2.0
                x_range = fig.layout.xaxis.range
                if x_range is not None:
                    x_min, x_max = x_range
                    x_center = (x_min + x_max) / 2
                    x_half = (x_max - x_min) / 2
                    if key == 'ArrowUp':
                        x_half /= zoom_factor
                    elif key == 'ArrowDown':
                        x_half *= zoom_factor
                    elif key == 'ArrowRight':
                        shift_val = x_half * 2 * move_factor
                        x_center += shift_val
                    elif key == 'ArrowLeft':
                        shift_val = x_half * 2 * move_factor
                        x_center -= shift_val
                    fig.update_xaxes(
                        range=[x_center - x_half, x_center + x_half],
                        rangeslider=dict(visible=rangeslider)
                    )
        elif event['event'] == 'keyup':
            if event.get('key', '') == 'y':
                fig.update_yaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', fixedrange=True
                )
                fig.update_xaxes(
                    mirror=True, ticks='outside', showline=True,
                    gridcolor='lightgrey', fixedrange=False,
                    rangeslider=dict(visible=rangeslider)
                )

    im_events = events.Event(source=fig, watched_events=['keydown', 'keyup'])
    im_events.prevent_default_action = True
    im_events.on_dom_event(interact_plot)

    _help_html = _widgets.HTML(
        '<div style="color: #888; font-size: 12px; padding: 5px;">'
        'Keyboard: ↑↓ zoom X | ←→ pan X | Alt+arrows for Y | Shift fine | Home reset | F toggle height</div>'
    )
    container_children = [fig, _help_html]

    if shift:
        x_step = x[1]-x[0]
        x_min = -1*x[len(x)//4]
        x_max = x[len(x)//4]
        slider = _widgets.FloatSlider(
            value=0,
            min=x_min,
            max=x_max,
            step=x_step,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            description='shift:',
            layout=_widgets.Layout(width='60%'),
        )
        if name is not None:
            drop_option = name
        else:
            drop_option = [str(i) for i in range(ntraces)]
        drop = _widgets.Dropdown(
            options=drop_option,
            value=drop_option[0],
            description='name:',
            disabled=False,
            layout=_widgets.Layout(width='150px'),
        )

        def update_plot(*args):
            idx = drop.index
            if resample:
                fig.hf_data[idx]['x'] = x + slider.value
                fig.reload_data()
            else:
                fig.data[idx].x = x + slider.value
        slider.observe(update_plot, 'value')
        container_children.append(_widgets.HBox([drop, slider]))
    container = _widgets.VBox(container_children)
    container.fig = fig
    return container

def _normalize_plot_arrays(arrays):
    if isinstance(arrays, (list, tuple)):
        groups = [_np.asarray(a).reshape(-1, _np.asarray(a).shape[-1]) for a in arrays]
    else:
        arr = _np.asarray(arrays)
        groups = [arr.reshape(-1, arr.shape[-1])]
    if not groups:
        raise ValueError("arrays must contain at least one array")
    return groups


def _normalize_plot_x(x, groups):
    if x is None:
        return [_np.arange(group.shape[-1]) for group in groups]
    x = _np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    for group in groups:
        if len(x) != group.shape[-1]:
            raise ValueError("x length must match every array's sample length")
    return [x for _ in groups]


def _normalize_plot_names(names, groups):
    if names is None:
        return [[str(i) if len(groups) == 1 else f"{i}-{j}" for j in range(group.shape[0])] for i, group in enumerate(groups)]
    if len(groups) == 1 and len(names) == groups[0].shape[0] and not any(isinstance(n, (list, tuple)) for n in names):
        return [list(names)]
    if len(names) != len(groups):
        raise ValueError("names length must match arrays length")
    normalized = []
    for group_idx, (group_name, group) in enumerate(zip(names, groups)):
        if isinstance(group_name, (list, tuple)):
            if len(group_name) != group.shape[0]:
                raise ValueError("nested names length must match trace count")
            normalized.append(list(group_name))
        elif group.shape[0] == 1:
            normalized.append([str(group_name)])
        else:
            normalized.append([f"{group_name}-{i}" for i in range(group.shape[0])])
    return normalized


def plot_multi(arrays, x: _np.ndarray = None, names: list = None, title: str = None,
               layout: str = "subplots", share_x: bool = True, yzoom=False,
               webgl=True, resample="auto"):
    """Plot several arrays either overlaid or in shared-x subplot rows.

    Use this to compare related signals on the same sample axis, such as CPA
    scores, raw traces, filtered traces, or intermediate metrics. Each input is
    normalized to ``(n_traces, n_samples)``; each row becomes one Plotly trace.

    Example:
        >>> fig = plot_multi(
        ...     [cpa_scores, raw_trace],
        ...     names=["CPA", "raw"],
        ...     layout="subplots",
        ... )
        >>> fig.fig

    Args:
        arrays: One 1-D/2-D array or a list of 1-D/2-D arrays. The last
            dimension is the sample axis.
        x: Optional shared 1-D x-axis. Its length must match every array's last
            dimension. Defaults to sample indices.
        names: Optional trace naming rules:
            - Single 2-D array: one name per row, e.g. ``["CPA", "raw"]``.
            - Multiple arrays: one name per array, e.g. ``["CPA", "raw"]``;
              names for multi-row arrays expand to ``CPA-0``, ``CPA-1``, ...
            - Nested names: one list per array, e.g.
              ``[["CPA0", "CPA1"], ["raw0"]]``.
        title: Figure title.
        layout: ``"subplots"`` places each input array in its own row;
            ``"overlay"`` plots all traces in one axes. Defaults to
            ``"subplots"``.
        share_x: Link x-axis zoom and pan across subplot rows. Only applies to
            ``layout="subplots"``. Defaults to True.
        yzoom: Allow y-axis zoom and pan. Defaults to False.
        webgl: Use Plotly ``Scattergl`` traces for faster rendering. Defaults
            to True.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it when
            total input size exceeds 100000 samples. Defaults to ``"auto"``.

    Returns:
        An ``ipywidgets.VBox`` containing the Plotly figure. The underlying
        figure is available as ``container.fig``.
    """
    if layout not in ("subplots", "overlay"):
        raise ValueError("layout must be 'subplots' or 'overlay'")

    groups = _normalize_plot_arrays(arrays)
    xs = _normalize_plot_x(x, groups)
    trace_names = _normalize_plot_names(names, groups)
    total_size = sum(group.size for group in groups)
    if resample == "auto":
        resample = total_size > 100000

    n_rows = len(groups) if layout == "subplots" else 1
    base_fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=share_x) if layout == "subplots" else _go.Figure()
    fig = FigureWidgetResampler(base_fig) if resample else _go.FigureWidget(base_fig)
    color_list = _px.colors.qualitative.Plotly
    scatter_cls = _go.Scattergl if webgl else _go.Scatter
    color_idx = 0

    for group_idx, group in enumerate(groups):
        row = group_idx + 1 if layout == "subplots" else None
        for trace_idx, series in enumerate(group):
            color = color_list[color_idx % len(color_list)]
            color_idx += 1
            trace = scatter_cls(
                name=trace_names[group_idx][trace_idx],
                opacity=0.8,
                line_color=color,
            )
            if resample:
                series = series if series.flags['C_CONTIGUOUS'] else _np.ascontiguousarray(series)
                if layout == "subplots":
                    fig.add_trace(trace, hf_x=xs[group_idx], hf_y=series, row=row, col=1)
                else:
                    fig.add_trace(trace, hf_x=xs[group_idx], hf_y=series)
            else:
                trace.x = xs[group_idx]
                trace.y = series
                if layout == "subplots":
                    fig.add_trace(trace, row=row, col=1)
                else:
                    fig.add_trace(trace)

    fig.update_layout(
        title=dict(text=title),
        plot_bgcolor="white",
        height=300 + 180 * n_rows,
        hovermode="x unified",
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
        gridcolor='lightgrey',
        fixedrange=not yzoom,
    )
    fig.update_traces(hoverinfo="name+x+y")

    help_html = _widgets.HTML(
        '<div style="color: #888; font-size: 12px; padding: 5px;">'
        'Use Plotly zoom/pan tools; subplot layout shares x-axis when share_x=True.</div>'
    )
    container = _widgets.VBox([fig, help_html])
    container.fig = fig
    return container


def plot_separate(y: _np.ndarray, x: _np.ndarray = None, name: list = None, resample="auto", yzoom=False):
    """Plot each row of one array in a separate shared-x subplot row.

    This is a convenience wrapper around ``plot_multi(..., layout="subplots")``
    for the common case where all rows come from one array.

    Args:
        y: 1-D or 2-D array. A 2-D array is plotted row by row.
        x: Optional shared 1-D x-axis. Defaults to sample indices.
        name: Optional trace names. Length must match the number of rows in
            ``y``.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it for
            large arrays. Defaults to ``"auto"``.
        yzoom: Allow y-axis zoom and pan. Defaults to False.

    Returns:
        An ``ipywidgets.VBox`` containing the Plotly figure. The underlying
        figure is available as ``container.fig``.
    """
    y = _np.asarray(y).reshape(-1, _np.asarray(y).shape[-1])
    arrays = [row for row in y]
    return plot_multi(arrays, x=x, names=name, layout="subplots", share_x=True, resample=resample, yzoom=yzoom)


def plot_seperate(y: _np.ndarray, x: _np.ndarray = None, resample="auto", yzoom=False):
    """Deprecated spelling of ``plot_separate``.

    Args:
        y (ndarray): 1-D/2-D array to plot row by row.
        x (ndarray, optional): Shared 1-D x-axis. Defaults to sample indices.
        resample (bool or str, optional): Use plotly-resampler, or ``"auto"`` for large arrays. Defaults to ``"auto"``.
        yzoom (bool, optional): Allow y-axis zoom. Defaults to False.

    Returns:
        ipywidgets.VBox: Widget returned by ``plot_separate``.
    """
    return plot_separate(y, x=x, resample=resample, yzoom=yzoom)


def plot_subfigure(y: _np.ndarray, x: _np.ndarray = None, name: list = None, resample="auto", yzoom=False):
    """Plot each row of one array in separate shared-x subplot rows.

    Args:
        y (ndarray): 1-D/2-D array. A 2-D array is plotted row by row.
        x (ndarray, optional): Shared 1-D x-axis. Defaults to sample indices.
        name (list, optional): Trace names. Length must match the number of rows in ``y``.
        resample (bool or str, optional): Use plotly-resampler, or ``"auto"`` for large arrays. Defaults to ``"auto"``.
        yzoom (bool, optional): Allow y-axis zoom. Defaults to False.

    Returns:
        ipywidgets.VBox: Widget returned by ``plot_separate``.
    """
    return plot_separate(y, x=x, name=name, resample=resample, yzoom=yzoom)


def plot_pattern(y: _np.ndarray, pattern_x: _np.ndarray, x: _np.ndarray = None, webgl=True, resample="auto", yzoom=False):
    """Plot a 1-D trace and highlight selected sample positions.

    Args:
        y: 1-D sample array.
        pattern_x: Integer sample indices to highlight.
        x: Optional x-axis for the full trace. Defaults to sample indices.
        webgl: Use Plotly ``Scattergl`` traces for faster rendering. Defaults
            to True.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it when
            ``y.size > 100000``. Defaults to ``"auto"``.
        yzoom: Allow y-axis zoom and pan. Defaults to False.

    Returns:
        An interactive Plotly ``FigureWidget``.
    """
    if resample == "auto":
        if y.size > 100000:
            resample = True
        else:
            resample = False
    if y.ndim > 1:
        raise ValueError("plot_pattern only supports 1D data input")
    pattern_x = _np.array(pattern_x)
    pattern = y[pattern_x]
    pattern_plot_x = pattern_x if x is None else _np.asarray(x)[pattern_x]
    fig = _go.FigureWidget()
    rangeslider = True
    if resample:
        fig = FigureWidgetResampler(fig)
        rangeslider = False
    if webgl:
        fig.add_trace(
            _go.Scattergl(y=y, x=x, opacity=0.8, name="trace"),

        )
        fig.add_trace(
            _go.Scattergl(y=pattern, x=pattern_plot_x, opacity=0.8, name="pattern"),

        )
    else:
        fig.add_trace(
            _go.Scatter(y=y, x=x, opacity=0.8, name="trace"),

        )
        fig.add_trace(
            _go.Scatter(y=pattern, x=pattern_plot_x, opacity=0.8, name="pattern"),

        )
    fig.update_layout(
        plot_bgcolor="white",
        height=400,
        hovermode="x unified"
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        rangeslider=dict(
            visible=rangeslider
        )
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        fixedrange=not yzoom
    )
    fig.update_traces(
        hoverinfo="name+x+y",
    )
    return fig


def plot_peak(y: _np.ndarray, peak_x: Union[_np.ndarray, int, list], x: _np.ndarray = None, webgl=True,  resample="auto", yzoom=False):
    """Plot a 1-D trace and mark peak sample positions.

    Args:
        y: 1-D sample array.
        peak_x: One sample index, or an array/list of sample indices, to mark as
            peaks.
        x: Optional x-axis for the full trace. Defaults to sample indices.
        webgl: Use Plotly ``Scattergl`` traces for faster rendering. Defaults
            to True.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it when
            ``y.size > 100000``. Defaults to ``"auto"``.
        yzoom: Allow y-axis zoom and pan. Defaults to False.

    Returns:
        An interactive Plotly ``FigureWidget``.
    """
    if resample == "auto":
        if y.size > 100000:
            resample = True
        else:
            resample = False
    if y.ndim > 1:
        raise ValueError("plot_peak only supports 1D data input")
    peak = y[peak_x]
    peak_plot_x = peak_x if x is None else _np.asarray(x)[peak_x]
    fig = _go.FigureWidget()
    rangeslider = True
    if resample:
        fig = FigureWidgetResampler(fig)
        rangeslider = False
    fig.update_layout(
        plot_bgcolor="white",
        height=400,
        hovermode="x unified"
    )
    if webgl:
        fig.add_trace(
            _go.Scattergl(y=y, x=x, opacity=0.8, name="trace"),
        )
        fig.add_trace(
            _go.Scattergl(y=peak, x=peak_plot_x, opacity=0.9,
                        mode="markers", marker_size=10, name="peaks"),
        )
    else:
        fig.add_trace(
            _go.Scatter(y=y, x=x, opacity=0.8, name="trace"),
        )
        fig.add_trace(
            _go.Scatter(y=peak, x=peak_plot_x, opacity=0.9,
                        mode="markers", marker_size=10, name="peaks"),
        )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        rangeslider=dict(
            visible=rangeslider
        )
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        fixedrange=not yzoom
    )
    fig.update_traces(
        hoverinfo="name+x+y",
    )
    return fig


def plot_heatmap(pos: _np.ndarray, y: _np.ndarray, x: _np.ndarray=None, resample="auto", yzoom=False, method='mean'):
    """Display scan-position leakage as a heatmap with linked trace preview.

    ``pos`` must describe a rectangular scan on one z-plane. The heatmap cell
    value is the maximum aggregated trace value for each x/y coordinate. Click a
    heatmap cell to update the trace preview for that scan position.

    Args:
        pos: Coordinate array with shape ``(n_traces, 3)``. Columns are x, y,
            and z positions.
        y: Trace samples with shape ``(n_traces, n_samples)``.
        x: Optional shared 1-D sample axis for trace previews. Defaults to
            sample indices.
        resample: Enable ``plotly-resampler``. Use ``"auto"`` to enable it for
            long traces. Defaults to ``"auto"``.
        yzoom: Allow y-axis zoom in the trace preview. Defaults to False.
        method: Aggregation for traces at the same x/y coordinate. Supported
            values are ``"mean"`` and ``"var"``. Defaults to ``"mean"``.

    Returns:
        None. The heatmap and trace preview are displayed in the active
        notebook output cell.
    """
    if resample == "auto":
        resample = y.shape[-1] > 100000
    if not method in ('mean', 'var'):
        raise ValueError("method only supports 'mean' and 'var'")
    if not pos.shape[1] == 3:
        raise ValueError("coordinates must contain x, y, z values")
    pos_z = pos[0, 2]
    if not _np.all(pos[:, 2] == pos_z):
        raise ValueError("scan coordinates z values are not on the same plane")
        
    xyz = _np.unique(pos, axis=0)
    pos_rows = _np.unique(xyz[:, 1])
    pos_cols = _np.unique(xyz[:, 0])
    scores = _np.zeros((len(pos_rows), len(pos_cols)), dtype=_np.float64)
    tsample = _np.zeros((len(pos_rows), len(pos_cols), y.shape[-1]), dtype=_np.float64)
    for i in range(len(pos_rows)):
        for j in range(len(pos_cols)):
            pos_x = pos_cols[j]
            pos_y = pos_rows[i]
            idx = _np.argwhere((pos[:, 0] == pos_x) * (pos[:, 1] == pos_y) *(pos[:, 2] == pos_z)).squeeze()
            subgroup_samples = y[idx]
            if method == 'mean':
                tsample[i,j] = _np.abs(_np.mean(subgroup_samples, axis=0))
            else:
                tsample[i,j] = _np.var(subgroup_samples, axis=0)
            scores[i, j] = _np.max(tsample[i,j])

    hovertext = list()
    for yi, yy in enumerate(pos_rows):
        hovertext.append(list())
        for xi, xx in enumerate(pos_cols):
            hovertext[-1].append('x: {}<br />y: {}<br />score: {:.2f}'.format(xx, yy, scores[yi, xi]))
    fig = _go.FigureWidget()
    fig.add_trace(
        _go.Heatmap(z=scores, x=pos_rows, y=pos_cols, hoverinfo='text', text=hovertext),
    )
    fig.update_layout(
        plot_bgcolor="white",
        height=500,
        width=500 
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    
    sfig =make_subplots(rows=2, cols=1)
    if resample:
        sfig = FigureWidgetResampler(sfig)
        sfig.add_trace(
            _go.Scattergl(opacity=0.8, name=method),
            hf_x=x, hf_y=tsample[0, 0],
            row=1, col=1,
        )
        sfig.add_trace(
            _go.Scattergl(opacity=0.8, name='raw-0'),
            hf_x=x, hf_y=y[0],
            row=2, col=1
        )
    else:
        sfig = _go.FigureWidget(sfig)
        sfig.add_trace(
            _go.Scattergl(x=x, y=tsample[0, 0], opacity=0.8, name=method),
            row=1, col=1,
        )
        sfig.add_trace(
            _go.Scattergl(x=x, y=y[0], opacity=0.8, name='raw-0'),
            row=2, col=1,
        )
    title = f'{pos_cols[0]},{pos_rows[0]},{pos_z}'
    sfig.update_layout(
        title=dict(text=title, x=0.5),
        plot_bgcolor="white",
        autosize=True,
        height=500,
        width=1000
    )
    
    sfig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey'
    )
    sfig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        fixedrange=not yzoom
    )

    def update_plot(trace, points, selector):
        pos_inds = points.point_inds[0]
        tpos = [pos_cols[pos_inds[1]], pos_rows[pos_inds[0]], pos_z]
        title = f'{tpos[0]},{tpos[1]},{pos_z}'
        idx = _np.argwhere((pos[:, 0] == tpos[0]) * (pos[:, 1] == tpos[1]) *(pos[:, 2] == pos_z)).squeeze()
        sfig.update_layout(
            title=dict(text=title, x=0.5)
        )
        if resample:
            sfig.hf_data[1]["name"] = f'raw-{idx[0]}'
            sfig.hf_data[0]["y"] = tsample[pos_inds[0], pos_inds[1]]
            sfig.hf_data[1]["y"] = y[idx[0]]
            sfig.reset_axes()
        else:
            sfig.data[1].name = f'raw-{idx[0]}'
            sfig.data[0].y = tsample[pos_inds[0], pos_inds[1]]
            sfig.data[1].y = y[idx[0]]
            
            
    fig.data[0].on_click(update_plot)
    display(_widgets.HBox([fig, sfig]))
    

def plot_spectrogram(y: _np.ndarray, fs=1.0, N=256, fslim=None):
    """Plot the spectrogram of a 1-D trace.

    Args:
        y: 1-D sample array.
        fs: Sampling frequency passed to ``scipy.signal.spectrogram``. Defaults
            to 1.0.
        N: FFT length passed as ``nfft``. Defaults to 256.
        fslim: Optional frequency range ``(fmin, fmax)`` to display.

    Returns:
        An interactive Plotly heatmap ``FigureWidget`` with time on the x-axis
        and frequency on the y-axis.
    """
    if y.ndim > 1:
        raise ValueError("plot_spectrogram only supports 1D data input")
    freqs, bins, Pxx = signal.spectrogram(y, fs, nfft=N)
    if fslim is not None:
        freq_slice = _np.where((freqs >= fslim[0]) & (freqs <= fslim[1]))
        freqs = freqs[freq_slice]
        Pxx = Pxx[freq_slice, :][0]
    fig = _go.FigureWidget()
    fig.add_trace(_go.Heatmap(x=bins, y=freqs, z=10*_np.log10(Pxx), colorscale='Jet'))
    fig.update_layout(
        plot_bgcolor="white",
        height=400,
        yaxis = dict(title = 'Frequency'),
        xaxis = dict(title = 'Time')
    )
    return fig



def _plot_step_score(y: _np.ndarray, guesses: list, steps: list, best_candidate: _np.ndarray, correct_key: _np.ndarray = None):
    """Display score evolution across output steps as Plotly subplots."""
    guess_values = list(guesses)
    step_values = list(steps)
    nb_sel_target = len(best_candidate)
    row = math.ceil(nb_sel_target / 4.0)
    titles = [f"index {i} candidate: 0x{best_candidate[i]:02x}" for i in range(nb_sel_target)]
    fig = make_subplots(rows=row, cols=4, subplot_titles=titles)

    for i in range(nb_sel_target):
        row_idx = i // 4 + 1
        col_idx = i % 4 + 1
        x_all = []
        y_all = []
        for guess_idx in range(len(guess_values)):
            x_all.extend(step_values + [None])
            y_all.extend(list(y[guess_idx, i, :]) + [None])
        fig.add_trace(
            _go.Scattergl(
                x=x_all,
                y=y_all,
                mode='lines',
                line=dict(color='lightgrey', width=1),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )
        best_idx = guess_values.index(best_candidate[i])
        best_color = 'green' if correct_key is None or best_candidate[i] == correct_key[i] else 'red'
        highlights = [(best_idx, best_color)]
        if correct_key is not None and best_candidate[i] != correct_key[i] and correct_key[i] in guess_values:
            highlights.append((guess_values.index(correct_key[i]), 'green'))
        for guess_idx, color in highlights:
            fig.add_trace(
                _go.Scattergl(
                    x=step_values,
                    y=y[guess_idx, i, :],
                    mode='lines',
                    name=f"0x{guess_values[guess_idx]:02x}",
                    line=dict(color=color, width=2.5),
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(title=dict(text='Step score'), height=300 * row, plot_bgcolor='white')
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, gridcolor='lightgrey')
    display(fig)


def _scatter_score(y: _np.ndarray, guesses: list, best_candidate: _np.ndarray, correct_key: _np.ndarray = None):
    """Display per-target guess scores as Plotly scatter subplots."""
    guess_values = list(guesses)
    nb_sel_target = len(best_candidate)
    row = math.ceil(nb_sel_target / 4.0)
    titles = [f"index {i} candidate: 0x{best_candidate[i]:02x}" for i in range(nb_sel_target)]
    fig = make_subplots(rows=row, cols=4, subplot_titles=titles)

    for i in range(nb_sel_target):
        row_idx = i // 4 + 1
        col_idx = i % 4 + 1
        best_idx = guess_values.index(best_candidate[i])
        fig.add_trace(
            _go.Scattergl(
                x=guess_values,
                y=y[:, i],
                mode='markers',
                marker=dict(color='lightgrey', size=5),
                showlegend=False,
            ),
            row=row_idx,
            col=col_idx,
        )
        if correct_key is not None and best_candidate[i] != correct_key[i]:
            fig.add_trace(
                _go.Scattergl(
                    x=[best_candidate[i]],
                    y=[y[best_idx, i]],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-up'),
                    name=f"best 0x{best_candidate[i]:02x}",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            if correct_key[i] in guess_values:
                correct_idx = guess_values.index(correct_key[i])
                fig.add_trace(
                    _go.Scattergl(
                        x=[correct_key[i]],
                        y=[y[correct_idx, i]],
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='star'),
                        name=f"correct 0x{correct_key[i]:02x}",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
        else:
            fig.add_trace(
                _go.Scattergl(
                    x=[best_candidate[i]],
                    y=[y[best_idx, i]],
                    mode='markers',
                    marker=dict(color='green', size=12, symbol='star'),
                    name=f"best 0x{best_candidate[i]:02x}",
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(title=dict(text='Score scatter'), height=300 * row, plot_bgcolor='white')
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, gridcolor='lightgrey')
    display(fig)


def _candidate_curve(y: _np.ndarray, guesses: list, best_candidate, target_idx: int, correct_key=None, x=None):
    """Build an interactive curve plot for one target's guess traces.

    Args:
        y (ndarray): Result array with shape ``(nb_guesses, nb_targets, nb_samples)``.
        guesses (list): Guess values corresponding to rows in ``y``.
        best_candidate: Rank-1 guess value for ``target_idx``.
        target_idx (int): Target byte/index to display.
        correct_key (ndarray, optional): Correct key values for green/red highlighting.
        x (ndarray, optional): Sample/time axis values. Defaults to ``range(nb_samples)``.

    Returns:
        ipywidgets.VBox: Plotly figure container with the underlying figure available as ``.fig``.
    """
    if target_idx < 0 or target_idx >= y.shape[1]:
        raise ValueError("target_idx out of range")
    if x is None:
        x = _np.arange(y.shape[-1])

    fig = _go.FigureWidget()
    y_idx = guesses.index(best_candidate)
    correct_idx = None
    if correct_key is not None and correct_key[target_idx] in guesses:
        correct_idx = guesses.index(correct_key[target_idx])

    for i, guess in enumerate(guesses):
        if i == y_idx or (correct_idx is not None and i == correct_idx):
            continue
        fig.add_trace(
            _go.Scattergl(
                x=x,
                y=y[i, target_idx, :],
                name=f"0x{guess:02x}",
                mode='lines',
                opacity=0.5,
                line=dict(color='lightgrey', width=1),
            )
        )

    highlight = []
    if correct_idx is not None and correct_idx != y_idx:
        highlight.append((correct_idx, 'green'))
    highlight.append((y_idx, 'green' if correct_key is None or best_candidate == correct_key[target_idx] else 'red'))
    for idx, color in highlight:
        fig.add_trace(
            _go.Scattergl(
                x=x,
                y=y[idx, target_idx, :],
                name=f"0x{guesses[idx]:02x}",
                mode='lines',
                opacity=1.0,
                line=dict(color=color, width=2.5),
            )
        )

    fig.update_layout(
        title=dict(text=f"index {target_idx} candidate: 0x{best_candidate:02x}"),
        plot_bgcolor="white",
        hovermode="x unified",
        height=500,
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        rangeslider=dict(visible=True),
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        gridcolor='lightgrey',
        fixedrange=True,
    )
    fig.update_traces(hoverinfo="name+x+y")
    container = _widgets.VBox([fig])
    container.fig = fig
    return container


