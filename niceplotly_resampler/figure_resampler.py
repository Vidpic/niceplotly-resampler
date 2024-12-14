from typing import Optional, Dict, Any 
import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from tsdownsample import MinMaxLTTBDownsampler as MinMaxLTTB
from functools import partial
from nicegui import ui

class FigureResampler:
    def __init__(
        self,
        figure: Optional[go.Figure] = go.Figure(),
        num_points: int = 1000,
        downsampler=MinMaxLTTB(),
    ):
        """
        A resampling Plotly Figure wrapper supporting dynamic updates and downsampling.
        """
        self.num_points = num_points
        self.downsampler = downsampler
        self.figure = figure
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.plot = None

        # Allow zooming in both x and y axes if no subplots are present
        self.figure.update_layout(dragmode="zoom", xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False))

    def add_trace(self, trace: Optional[BaseTraceType] = None, row: Optional[int] = None, col: Optional[int] = None, **kwargs) -> go.Figure:
        if trace is None:
            trace = go.Scattergl(connectgaps=False, **kwargs)

        trace_name = trace.name if trace.name else f"Trace {len(self.traces) + 1}"
        x_data = np.asarray(trace.x)
        y_data = np.asarray(trace.y)

        trace.name = f'<span style="color:orange;">[R]</span> {trace_name}'

        # Add trace to the correct subplot if rows and columns are specified
        if row is not None and col is not None:
            self.figure.add_trace(trace, row=row, col=col)
            # Fix y-axis for subplots
            self.figure.update_yaxes(fixedrange=True, row=row, col=col)
        else:
            self.figure.add_trace(trace)

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
        """
        self.figure.update_layout(**kwargs)

    def _format_bin_size(self, bin_size: int) -> str:
        """
        Format the bin size with appropriate units (k, M, G, etc.).
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

    def _resample_all_traces(self) -> None:
        for i, (trace_name, trace_info) in enumerate(self.traces.items()):
            x = trace_info["x"]
            y = trace_info["y"]

            x_range = self.figure.layout.xaxis.range or (x.min(), x.max())

            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_filtered = x[mask]
            y_filtered = y[mask]

            total_points = x_filtered.size

            if total_points > self.num_points:
                indices = self.downsampler.downsample(x_filtered, y_filtered, n_out=self.num_points)
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

    def show(self, options: Optional[dict] = None) -> None:
        """
        Show the figure initially. Sets up the Plotly figure in NiceGUI and attaches event handlers.
        """
        if options is None:
            options = {}
        self._resample_all_traces()
        fig_dict = self.figure.to_dict()
        fig_dict["config"] = options

        if self.plot:
            # Update the existing plot
            self.plot.figure = self.figure
            self.plot.update()
        else:
            # Create a new plot
            self.plot = ui.plotly(fig_dict)
            self.plot.on("plotly_relayout", partial(self._on_relayout))
            self.plot.on("plotly_doubleclick", partial(self._on_doubleclick))

    def reset(self) -> None:
        """
        Reset the figure layout and resample all traces.
        """
        self.traces = {}
        self.figure.data = []

    async def _on_relayout(self, event: Any) -> None:
        args = event.args
        updated_subplots = {}

        for k, v in args.items():
            if ".range[" not in k:
                continue

            axis_part, range_part = k.split(".range[")
            axis_part = axis_part.strip()

            row, col = self._axis_to_subplot(axis_part)

            if (row, col) not in updated_subplots:
                updated_subplots[(row, col)] = {"x_range": None, "y_range": None}

            idx_str = range_part.rstrip("]")
            idx = int(idx_str)
            val = float(v)

            is_x_axis = axis_part.startswith('xaxis') or axis_part == 'xaxis'

            if is_x_axis:
                curr_range = updated_subplots[(row, col)]["x_range"]
                if curr_range is None:
                    curr_range = [None, None]
                curr_range[idx] = val
                updated_subplots[(row, col)]["x_range"] = tuple(curr_range) if None not in curr_range else curr_range
            else:
                curr_range = updated_subplots[(row, col)]["y_range"]
                if curr_range is None:
                    curr_range = [None, None]
                curr_range[idx] = val
                updated_subplots[(row, col)]["y_range"] = tuple(curr_range) if None not in curr_range else curr_range

        for (row, col), rng_info in updated_subplots.items():
            x_r = rng_info["x_range"]
            y_r = rng_info["y_range"]

            if x_r and None not in x_r:
                self.figure.update_layout(xaxis_range=x_r)
            if y_r and None not in y_r:
                self.figure.update_layout(yaxis_range=y_r)

        self._resample_all_traces()
        self.plot.figure = self.figure
        self.plot.update()
        
    async def _on_doubleclick(self, event: Any) -> None:
        """
        Handle double-click events to reset the figure layout and resample all traces.
        """
        self.update_layout(xaxis_range=None, yaxis_range=None)
        self._resample_all_traces()
        self.plot.figure = self.figure
        self.plot.update()

    def _axis_to_subplot(self, axis_name: str) -> tuple:
        """
        Determine the subplot row and column from the axis name.
        """
        if axis_name == "xaxis":
            axis_type, axis_num = 'x', 1
        elif axis_name == "yaxis":
            axis_type, axis_num = 'y', 1
        elif axis_name.startswith('xaxis'):
            axis_type = 'x'
            axis_num = int(axis_name[5:])
        elif axis_name.startswith('yaxis'):
            axis_type = 'y'
            axis_num = int(axis_name[5:])
        else:
            axis_type, axis_num = 'x', 1

        N = axis_num - 1
        row = (N // 1) + 1
        col = (N % 1) + 1
        return row, col