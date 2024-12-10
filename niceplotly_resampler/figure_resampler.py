from typing import Optional, Tuple, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType
from tsdownsample import (
    MinMaxLTTBDownsampler as MinMaxLTTB,
    MinMaxDownsampler as MinMax,
    M4Downsampler as M4,
    LTTBDownsampler as LTTB,
)
from .gap_handler import MedDiffGapHandler
from .nicegui_plotly_handler import show, update


class FigureResampler:
    def __init__(
        self,
        num_points: int = 1000,
        webgl: bool = False,
        downsampler=MinMaxLTTB(),
        gap_handler=MedDiffGapHandler(),
    ):
        """
        A resampling Plotly Figure wrapper supporting multiple subplots, gap handling, and downsampling.
        """
        self.num_points = num_points
        self.webgl = webgl
        self.downsampler = downsampler
        self.gap_handler = gap_handler

        self.figure = go.Figure()
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.rows = 1
        self.cols = 1
        self.subplot_ranges: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]] = {}
        self.subplot_mode = False
        self.plot = None
        self.shared_xaxes = False
        self.shared_yaxes = False

        self.figure.update_layout(dragmode="zoom")
        self.figure.update_xaxes(fixedrange=False)
        self.figure.update_yaxes(fixedrange=False)

    def _init_subplot_range(
        self, row: int, col: int, x_data: np.ndarray, y_data: np.ndarray
    ) -> None:
        if (row, col) not in self.subplot_ranges:
            x_range = (float(np.min(x_data)), float(np.max(x_data))) if x_data.size else (0, 100)
            y_range = (float(np.min(y_data)), float(np.max(y_data))) if y_data.size else (0, 1)

            self.subplot_ranges[(row, col)] = {
                "x_range": x_range,
                "y_range": y_range,
                "initial_x_range": x_range,
                "initial_y_range": y_range,
            }

    def add_trace(
        self,
        trace: Optional[BaseTraceType] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        secondary_y: Optional[bool] = None,
        exclude_empty_subplots: bool = False,
        **kwargs,
    ) -> go.Figure:
        row_arg = row if row is not None else 1
        col_arg = col if col is not None else 1
        scatter_cls = go.Scattergl if self.webgl else go.Scatter

        if trace is None or isinstance(trace, dict):
            trace = scatter_cls(connectgaps=False, **kwargs)
        else:
            trace.connectgaps = False
            for k, v in kwargs.items():
                setattr(trace, k, v)

        trace_name = trace.name if trace.name else f"Trace {len(self.traces) + 1}"
        x_data = np.asarray(trace.x)
        y_data = np.asarray(trace.y)

        self._init_subplot_range(row_arg, col_arg, x_data, y_data)

        # Initially mark with [R]; will finalize name in _resample_all_traces()
        trace.name = f'<span style="color:orange;">[R]</span> {trace_name}'

        trace_args = {
            'secondary_y': secondary_y,
            'exclude_empty_subplots': exclude_empty_subplots
        }
        if self.subplot_mode:
            trace_args['row'] = row_arg
            trace_args['col'] = col_arg

        self.figure.add_trace(trace, **trace_args)

        self.traces[trace_name] = {
            "x": x_data,
            "y": y_data,
            "row": row_arg,
            "col": col_arg,
            "original_name": trace_name,
        }

        return self.figure

    def update_layout(self, *args, **kwargs) -> go.Figure:
        self.figure.update_layout(*args, **kwargs)
        return self.figure

    def make_subplots(
        self,
        rows: int = 1,
        cols: int = 1,
        shared_xaxes: bool = False,
        shared_yaxes: bool = False,
        start_cell: str = "top-left",
        vertical_spacing: Optional[float] = None,
        horizontal_spacing: Optional[float] = None,
        subplot_titles: Optional[list] = None,
        specs: Optional[list] = None,
        column_widths: Optional[list] = None,
        row_heights: Optional[list] = None,
        print_grid: bool = False,
        x_title: Optional[str] = None,
        y_title: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        **kwargs,
    ) -> go.Figure:
        self.figure = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            start_cell=start_cell,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            subplot_titles=subplot_titles,
            specs=specs,
            column_widths=column_widths,
            row_heights=row_heights,
            print_grid=print_grid,
            x_title=x_title,
            y_title=y_title,
            figure=figure,
            **kwargs,
        )
        self.subplot_mode = True
        self.rows = rows
        self.cols = cols
        self.shared_xaxes = shared_xaxes
        self.shared_yaxes = shared_yaxes
        return self.figure

    def _insert_nans_for_gaps(
        self, x: np.ndarray, y: np.ndarray, gap_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        gap_indices = np.where(gap_mask)[0]
        if gap_indices.size == 0:
            return x, y

        x_new, y_new = [], []
        start_idx = 0
        for gap_idx in gap_indices:
            if gap_idx > start_idx:
                x_new.extend(x[start_idx:gap_idx])
                y_new.extend(y[start_idx:gap_idx])
                x_new.append(np.nan)
                y_new.append(np.nan)
                start_idx = gap_idx
            else:
                start_idx = gap_idx

        x_new.extend(x[start_idx:])
        y_new.extend(y[start_idx:])
        return np.array(x_new), np.array(y_new)

    def _resample_all_traces(self) -> None:
        updated_traces = []
        all_y_values = []

        for trace_name, trace_info in self.traces.items():
            row, col = trace_info["row"], trace_info["col"]
            subplot_info = self.subplot_ranges[(row, col)]
            x_range = subplot_info["x_range"]
            y_range = subplot_info["y_range"]

            x = trace_info["x"]
            y = trace_info["y"]

            x_start, x_end = x_range
            y_start, y_end = y_range
            mask = (x >= x_start) & (x <= x_end) & (y >= y_start) & (y <= y_end)
            x_filtered = x[mask]
            y_filtered = y[mask]
            total_points_in_range = x_filtered.size

            if total_points_in_range > self.num_points:
                indices = self.downsampler.downsample(x_filtered, y_filtered, n_out=self.num_points)
                x_filtered = x_filtered[indices]
                y_filtered = y_filtered[indices]

            bin_size = max(total_points_in_range // self.num_points, 1)
            if bin_size == 1:
                updated_name = trace_info["original_name"]
            else:
                updated_name = (
                    f'<span style="color:orange;">[R]</span> {trace_info["original_name"]} '
                    f'<span style="color:orange;">~{self._format_bin_size(bin_size)}</span>'
                )

            if self.gap_handler is not None and x_filtered.size > 1:
                gap_mask = self.gap_handler._get_gap_mask(x_filtered)
                if gap_mask is not None and np.any(gap_mask):
                    x_filtered, y_filtered = self._insert_nans_for_gaps(x_filtered, y_filtered, gap_mask)

            if x_filtered.size > 0:
                # Filter out NaNs for axis range calculation
                all_y_values.append(y_filtered[~np.isnan(y_filtered)])

            scatter_cls = go.Scattergl if self.webgl else go.Scatter
            seg_trace = scatter_cls(
                x=x_filtered, y=y_filtered, name=updated_name, connectgaps=False
            )
            updated_traces.append((seg_trace, row, col))

        self.figure.data = []
        for trace, row, col in updated_traces:
            if self.subplot_mode:
                self.figure.add_trace(trace, row=row, col=col)
            else:
                self.figure.add_trace(trace)

        # Adjust axis range and margins if we have data
        if all_y_values:
            all_y_combined = np.concatenate(all_y_values)
            y_min = float(np.min(all_y_combined))
            y_max = float(np.max(all_y_combined))
            self.figure.update_yaxes(range=[y_min, y_max])
            self.figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    def _format_bin_size(self, bin_size: int) -> str:
        if bin_size >= 10**12:
            return f"{bin_size // 10**12}T"
        elif bin_size >= 10**9:
            return f"{bin_size // 10**9}G"
        elif bin_size >= 10**6:
            return f"{bin_size // 10**6}M"
        elif bin_size >= 10**3:
            return f"{bin_size // 10**3}k"
        else:
            return str(bin_size)


    def _axis_to_subplot(self, axis_name: str) -> Tuple[int, int]:
        if axis_name == "xaxis":
            axis_type, axis_num = 'x', 1
        elif axis_name == "yaxis":
            axis_type, axis_num = 'y', 1
        elif axis_name.startswith('xaxis'):
            axis_type = 'x'
            axis_num = int(axis_name[5:])  # after 'xaxis'
        elif axis_name.startswith('yaxis'):
            axis_type = 'y'
            axis_num = int(axis_name[5:])
        else:
            axis_type, axis_num = 'x', 1

        N = axis_num - 1
        row = (N // self.cols) + 1
        col = (N % self.cols) + 1
        return (row, col)

    def show(self, options: Optional[dict] = None) -> None:
        show(self, options)

    def update(self) -> None:
        update(self)

    def reset(self) -> None:
        self.figure = go.Figure()
        self.traces.clear()
        self.subplot_ranges.clear()
        self.subplot_mode = False
        self.rows = 1
        self.cols = 1
        self.shared_xaxes = False
        self.shared_yaxes = False

        self.figure.update_layout(dragmode="zoom")
        self.figure.update_xaxes(fixedrange=False)
        self.figure.update_yaxes(fixedrange=False)
