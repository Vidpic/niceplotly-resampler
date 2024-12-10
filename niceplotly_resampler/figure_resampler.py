from typing import Optional, Tuple, Dict, Any
from nicegui import ui
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType
from plotly.graph_objs._figure import Figure

class FigureResampler:
    def __init__(self, num_points: int = 1000, webgl: bool = False):
        """
        Initialize the PlotlyResampler wrapper around a Plotly `go.Figure`.

        Parameters
        ----------
        num_points : int, default=1000
            The target number of points to downsample each trace to.
        webgl : bool, default=False
            Whether to use ScatterGL (True) or Scatter (False) for rendering.
        """
        self.figure: go.Figure = go.Figure()
        self.num_points: int = num_points
        self.webgl: bool = webgl
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.x_range: Tuple[float, float] = (0, 100)
        self.subplot_mode: bool = False
        self.plot = None

    def add_trace(
        self,
        trace: Optional[BaseTraceType] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        secondary_y: Optional[bool] = None,
        exclude_empty_subplots: bool = False,
        **kwargs
    ) -> Figure:
        """
        Add a trace to the figure.

        add_trace(
            trace: BaseTraceType or dict or None = None,
            row: int or str or None = None,
            col: int or str or None = None,
            secondary_y: bool or None = None,
            exclude_empty_subplots: bool = False,
            **kwargs
        ) -> Figure

        Adds a trace to the figure.

        Parameters
        ----------
        trace : BaseTraceType, dict, or None, default=None
            An instance of a Plotly trace (e.g. go.Scatter) or a dict describing a trace.
            If None, a trace will be constructed from the kwargs provided.
        row : int, str, or None, default=None
            Subplot row index (starting from 1) for the trace to be added. Only valid if the figure
            was created using `make_subplots`. If 'all', applies to all rows.
        col : int, str, or None, default=None
            Subplot column index (starting from 1) for the trace to be added. Only valid if the figure
            was created using `make_subplots`. If 'all', applies to all columns.
        secondary_y : bool or None, default=None
            If True, associate this trace with the secondary y-axis of the subplot at the specified row and col.
        exclude_empty_subplots : bool, default=False
            If True, the trace will not be added to subplots that don't already have axes.
        **kwargs
            Additional keyword arguments to be passed to the trace constructor (e.g., x, y, name, mode).

        Returns
        -------
        Figure
            The Figure object on which this method was called.

        Raises
        ------
        ValueError
            If using subplots and either `row` or `col` is not specified when subplot_mode is True.

        Examples
        --------
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]))
        Figure(...)

        Adding traces to subplots:
        >>> fig = make_subplots(rows=2, cols=1)
        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=1)
        Figure(...)
        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=2, col=1)
        Figure(...)
        """
        row_arg = row
        col_arg = col
        scatter_cls = go.Scattergl if self.webgl else go.Scatter

        # If no trace is provided, construct one from kwargs
        if trace is None or isinstance(trace, dict):
            trace = scatter_cls(**kwargs)
        else:
            # Update provided trace with any additional kwargs
            for k, v in kwargs.items():
                setattr(trace, k, v)

        trace_name = trace.name if trace.name else f"Trace {len(self.traces) + 1}"
        x_data = np.array(trace.x)
        bin_size = max(len(x_data) // self.num_points, 1)
        formatted_name = (
            f'<span style="color:orange;">[R]</span> {trace_name} '
            f'<span style="color:orange;">~{bin_size}</span>'
        )
        trace.name = formatted_name

        if self.subplot_mode and (row_arg is None or col_arg is None):
            raise ValueError("row and col must be specified when using subplots.")

        if self.subplot_mode:
            self.figure.add_trace(trace, row=row_arg, col=col_arg, secondary_y=secondary_y, exclude_empty_subplots=exclude_empty_subplots)
        else:
            self.figure.add_trace(trace, secondary_y=secondary_y, exclude_empty_subplots=exclude_empty_subplots)

        self.traces[trace_name] = {
            "data": pl.DataFrame({"x": trace.x, "y": trace.y}),
            "row": row_arg,
            "col": col_arg,
            "original_name": trace_name,
        }
        return self.figure

    def update_layout(self, *args, **kwargs) -> Figure:
        """
        Update the layout of the wrapped figure.

        Parameters
        ----------
        *args : Any
            Variable length argument list passed to `figure.update_layout`.
        **kwargs : Any
            Arbitrary keyword arguments accepted by `figure.update_layout`.

        Returns
        -------
        Figure
            The updated Figure object.
        """
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
        **kwargs
    ) -> Figure:
        """
        Create a figure with subplots.

        make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=False,
            shared_yaxes=False,
            start_cell="top-left",
            vertical_spacing=None,
            horizontal_spacing=None,
            subplot_titles=None,
            specs=None,
            column_widths=None,
            row_heights=None,
            print_grid=False,
            x_title=None,
            y_title=None,
            figure=None,
            **kwargs
        )

        Returns a figure with subplots.

        Parameters
        ----------
        rows : int, default=1
            Number of rows in the subplot grid.
        cols : int, default=1
            Number of columns in the subplot grid.
        shared_xaxes : bool, default=False
            Assign shared x axes.
        shared_yaxes : bool, default=False
            Assign shared y axes.
        start_cell : str, default="top-left"
            Choose the starting cell in the subplot grid.
        vertical_spacing : float or None, default=None
            Space between subplot rows.
        horizontal_spacing : float or None, default=None
            Space between subplot columns.
        subplot_titles : list or None, default=None
            Title of each subplot as a list in row-major order.
        specs : list or None, default=None
            Per-subplot specifications (e.g. 'xy', 'polar', 'scene', etc.).
        column_widths : list or None, default=None
            Widths of each column of subplots.
        row_heights : list or None, default=None
            Heights of each row of subplots.
        print_grid : bool, default=False
            If True, prints a string representation of the subplot grid.
        x_title : str or None, default=None
            Title placed below the bottom-most subplot.
        y_title : str or None, default=None
            Title placed to the left of the left-most subplot.
        figure : Figure or None, default=None
            If provided, subplots are added to this figure, otherwise a new one is created.
        **kwargs
            Additional keyword arguments passed to `plotly.subplots.make_subplots`.

        Returns
        -------
        Figure
            The Figure object with the specified subplots.

        Notes
        -----
        After calling this method, `self.subplot_mode` is set to True.
        """
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
            **kwargs
        )
        self.subplot_mode = True
        return self.figure

    def resample_trace(
        self,
        trace_data: pl.DataFrame,
        x_start: float,
        x_end: float
    ) -> Tuple[pl.DataFrame, int]:
        """
        Resample a single trace's data to the desired number of points in the given x-range.

        Parameters
        ----------
        trace_data : polars.DataFrame
            A dataframe containing columns "x" and "y".
        x_start : float
            The start of the x-axis range.
        x_end : float
            The end of the x-axis range.

        Returns
        -------
        (resampled_data: polars.DataFrame, total_points_in_range: int)
            A tuple where `resampled_data` is the downsampled dataframe and `total_points_in_range` 
            is the count of points in the specified range before downsampling.
        """
        filtered_data = trace_data.filter(
            (trace_data["x"] >= x_start) & (trace_data["x"] <= x_end)
        )
        total_points_in_range = filtered_data.shape[0]
        if total_points_in_range <= self.num_points:
            return filtered_data, total_points_in_range

        indices = np.linspace(0, total_points_in_range - 1, self.num_points).astype(int)
        return filtered_data[indices], total_points_in_range

    def resample_all_traces(self, x_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Resample all traces in the figure using the specified x_range or the current default x_range.

        Parameters
        ----------
        x_range : tuple of float, optional
            (x_start, x_end) specifying the x-axis range to resample over.
            If None, uses self.x_range.

        Returns
        -------
        None
        """
        x_start, x_end = x_range if x_range else self.x_range
        resampled_traces = []
        for trace_name, trace_info in self.traces.items():
            trace_data = trace_info["data"]
            resampled_data, total_points_in_range = self.resample_trace(
                trace_data, x_start, x_end
            )
            bin_size = max(total_points_in_range // self.num_points, 1)
            updated_name = (
                f'<span style="color:orange;">[R]</span> {trace_info["original_name"]} '
                f'<span style="color:orange;">~{bin_size}</span>'
            )
            scatter_cls = go.Scattergl if self.webgl else go.Scatter
            resampled_trace = scatter_cls(
                x=resampled_data["x"].to_numpy(),
                y=resampled_data["y"].to_numpy(),
                name=updated_name,
            )
            if self.subplot_mode:
                resampled_traces.append(
                    (resampled_trace, trace_info["row"], trace_info["col"])
                )
            else:
                resampled_traces.append((resampled_trace, None, None))

        self.figure.data = []
        for trace, row, col in resampled_traces:
            if row is not None and col is not None:
                self.figure.add_trace(trace, row=row, col=col)
            else:
                self.figure.add_trace(trace)

    async def on_relayout(self, event: Any) -> None:
        """
        Handle the `plotly_relayout` event to dynamically resample data based on the new x-axis range.

        Parameters
        ----------
        event : Any
            The event object from the frontend, containing updated layout information.

        Returns
        -------
        None
        """
        args = event.args
        x_start = float(args.get("xaxis.range[0]", self.x_range[0]))
        x_end = float(args.get("xaxis.range[1]", self.x_range[1]))
        self.resample_all_traces((x_start, x_end))
        self.plot.figure = self.figure
        self.plot.update()

    async def on_doubleclick(self, event: Any) -> None:
        """
        Handle the `plotly_doubleclick` event to reset the plot to the initial x-range.

        Parameters
        ----------
        event : Any
            The event object from the frontend.

        Returns
        -------
        None
        """
        self.resample_all_traces(self.x_range)
        self.plot.figure = self.figure
        self.plot.update()

    def show(self, options: Optional[dict] = None) -> None:
        """
        Create or update the NiceGUI plotly plot in the UI.

        Parameters
        ----------
        options : dict, optional
            A dictionary of Plotly configuration options (e.g., {"displayModeBar": False}).

        Returns
        -------
        None
        """
        if options is None:
            options = {}
        self.resample_all_traces()
        fig_dict = self.figure.to_dict()
        fig_dict["config"] = options
        if self.plot:
            self.plot.update(fig_dict)
        else:
            self.plot = ui.plotly(fig_dict)
            self.plot.on("plotly_relayout", self.on_relayout)
            self.plot.on("plotly_doubleclick", self.on_doubleclick)