from functools import partial
import numpy as np
from nicegui import ui
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from tsdownsample import (
    MinMaxDownsampler as MinMax,
    MinMaxLTTBDownsampler as MinMaxLTTB,
    LTTBDownsampler as LTTB,
    M4Downsampler as M4,
)
from typing import Optional, Dict, Any, Union
from .event_handlers import _on_relayout, _on_doubleclick


class FigureResampler:
    def __init__(
        self,
        figure: Optional[go.Figure] = go.Figure(),
        num_points: int = 1000,
        downsampler: Union[MinMax, M4, LTTB, MinMaxLTTB] = MinMaxLTTB(),
        parallel: bool = True,
    ):
        """
        A resampling Plotly Figure wrapper supporting dynamic updates and downsampling.

        Args:
            figure: The Plotly figure to wrap.
            num_points: The number of points to display in the figure.
            downsampler: The downsampler to use for resampling.
            parallel: Whether to use parallel processing for downsampling.
        """
        self.num_points = num_points
        self.downsampler = downsampler
        self.parallel = parallel
        self.figure = figure
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.plot = None

        self.figure.update_layout(
            dragmode="zoom", xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False)
        )

    def add_trace(
        self,
        trace: Optional[BaseTraceType] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        **kwargs,
    ) -> go.Figure:
        """
        Add a trace to the figure with optional row and column specification.

        Args:
            trace: The trace to add to the figure.
            row: The row of the subplot to add the trace to.
            col: The column of the subplot to add the trace to.
            **kwargs: Additional keyword arguments to pass to the trace.

        Returns:
            go.Figure: The updated figure with the added trace.
        """
        if trace is None:
            trace = go.Scattergl(connectgaps=False, **kwargs)

        trace_name = trace.name if trace.name else f"Trace {len(self.traces) + 1}"
        x_data = np.asarray(trace.x)
        y_data = np.asarray(trace.y)

        trace.name = f'<span style="color:orange;">[R]</span> {trace_name}'

        if row is not None and col is not None:
            self.figure.add_trace(trace, row=row, col=col)
            self.figure.update_yaxes(fixedrange=True, row=row, col=col)
        else:
            self.figure.add_trace(trace)

        # Ensure uirevision is set to preserve UI state for example when disabling traces
        self.figure.update_layout(uirevision=True)

        self.traces[trace_name] = {
            "x": x_data,
            "y": y_data,
            "row": row,
            "col": col,
            "original_name": trace_name,
        }

        return self.figure

    def update_layout(self, **kwargs) -> None:
        """
        Update the layout of the figure with the provided arguments.

        Args:
            **kwargs: Additional keyword arguments to pass to the figure layout.
        """
        self.figure.update_layout(**kwargs)

    def update(self, figure_resampler: "FigureResampler") -> None:
        """
        Update the figure with the provided FigureResampler.

        Args:
            figure_resampler: The FigureResampler to change the figure to.
        """
        self.figure = figure_resampler.figure
        self.traces = figure_resampler.traces

    def reset(self) -> None:
        """
        Reset the figure by removing all traces.
        """
        self.traces = {}
        self.figure.data = []

    def show(self, options: Optional[dict] = None) -> ui.plotly:
        """
        Create or update the nicegui plot with the current figure.

        Args:
            options: Additional options to pass to the plot

        Returns:
            ui.plotly: The nicegui plot object.
        """
        if options is None:
            options = {}
        self._resample_all_traces()
        fig_dict = self.figure.to_dict()
        fig_dict["config"] = options

        if self.plot:
            self.plot.figure = self.figure
            self.plot.update()
        else:
            self.plot = ui.plotly(fig_dict)
            self.plot.on("plotly_relayout", partial(_on_relayout, self))
            self.plot.on("plotly_doubleclick", partial(_on_doubleclick, self))

        return self.plot

    def _resample_all_traces(self) -> None:
        """
        Resample all traces in the figure
        """
        for i, (trace_name, trace_info) in enumerate(self.traces.items()):
            x = trace_info["x"]
            y = trace_info["y"]

            x_range = self.figure.layout.xaxis.range or (x.min(), x.max())

            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_filtered = x[mask]
            y_filtered = y[mask]

            total_points = x_filtered.size

            if total_points > self.num_points:
                indices = self.downsampler.downsample(
                    x_filtered,
                    y_filtered,
                    n_out=self.num_points,
                    parallel=self.parallel,
                )
                x_filtered = x_filtered[indices]
                y_filtered = y_filtered[indices]

            bin_size = max(total_points // self.num_points, 1)
            formatted_bin_size = self._format_bin_size(bin_size)

            self.figure.data[i].x = x_filtered
            self.figure.data[i].y = y_filtered

            self.figure.data[i].name = (
                f'<span style="color:orange;">[R]</span> {trace_info["original_name"]} '
                f'<span style="color:orange;">~{formatted_bin_size}</span>'
            )

    def _format_bin_size(self, bin_size: int) -> str:
        """
        Format the bin size with appropriate units (k, M, G, etc.).

        Args:
            bin_size: The bin size to format.

        Returns:
            The formatted bin size as a string.
        """
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

    def _axis_to_subplot(self, axis_name: str) -> tuple:
        """
        Determine the subplot row and column from the axis name.

        Args:
            axis_name: The name of the axis.

        Returns:
            tuple: The row and column of the subplot.
        """
        if axis_name == "xaxis":
            axis_type, axis_num = "x", 1
        elif axis_name == "yaxis":
            axis_type, axis_num = "y", 1
        elif axis_name.startswith("xaxis"):
            axis_type = "x"
            axis_num = int(axis_name[5:])
        elif axis_name.startswith("yaxis"):
            axis_type = "y"
            axis_num = int(axis_name[5:])
        else:
            axis_type, axis_num = "x", 1

        N = axis_num - 1
        row = (N // 1) + 1
        col = (N % 1) + 1
        return row, col
