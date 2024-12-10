from functools import partial
from typing import Optional, Any
from nicegui import ui


async def _on_relayout(fig_resampler, event: Any) -> None:
    args = event.args
    updated_subplots = {}

    # Parse updated ranges from event
    for k, v in args.items():
        if ".range[" not in k:
            continue  # Skip irrelevant keys

        axis_part, range_part = k.split(".range[")
        axis_part = axis_part.strip()

        # Use the updated _axis_to_subplot method directly
        row, col = fig_resampler._axis_to_subplot(axis_part)

        if (row, col) not in updated_subplots:
            updated_subplots[(row, col)] = {"x_range": None, "y_range": None}

        idx_str = range_part.rstrip("]")
        idx = int(idx_str)
        val = float(v)

        # Determine if this is an xaxis or yaxis by checking axis_part
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

    # Apply updated ranges to subplot_ranges
    x_range_to_set = None
    y_range_to_set = None
    for (r, c), rng_info in updated_subplots.items():
        subplot_current = fig_resampler.subplot_ranges[(r, c)]
        x_r = rng_info["x_range"]
        y_r = rng_info["y_range"]

        if x_r and None not in x_r:
            subplot_current["x_range"] = x_r
            if fig_resampler.shared_xaxes:
                x_range_to_set = x_r
        if y_r and None not in y_r:
            subplot_current["y_range"] = y_r
            if fig_resampler.shared_yaxes:
                y_range_to_set = y_r

    # Propagate shared axes if needed
    if fig_resampler.shared_xaxes and x_range_to_set is not None:
        for key in fig_resampler.subplot_ranges:
            fig_resampler.subplot_ranges[key]["x_range"] = x_range_to_set
    if fig_resampler.shared_yaxes and y_range_to_set is not None:
        for key in fig_resampler.subplot_ranges:
            fig_resampler.subplot_ranges[key]["y_range"] = y_range_to_set

    # Resample and update figure using the private method
    fig_resampler._resample_all_traces()
    fig_resampler.plot.figure = fig_resampler.figure
    fig_resampler.plot.update()


async def _on_doubleclick(fig_resampler, event: Any) -> None:
    # Reset to initial ranges
    first_subplot = (1, 1)
    initial = fig_resampler.subplot_ranges[first_subplot]
    initial_x_range = initial["initial_x_range"]
    initial_y_range = initial["initial_y_range"]

    apply_x = fig_resampler.shared_xaxes
    apply_y = fig_resampler.shared_yaxes

    for rng in fig_resampler.subplot_ranges.values():
        rng["x_range"] = initial_x_range if apply_x else rng["initial_x_range"]
        rng["y_range"] = initial_y_range if apply_y else rng["initial_y_range"]

    # Resample and update figure using the private method
    fig_resampler._resample_all_traces()

    # Update axis ranges
    for (row, col), rng in fig_resampler.subplot_ranges.items():
        axis_num = (row - 1) * fig_resampler.cols + col
        xaxis_name = "xaxis" if axis_num == 1 else f"xaxis{axis_num}"
        yaxis_name = "yaxis" if axis_num == 1 else f"yaxis{axis_num}"
        fig_resampler.figure.update_xaxes(range=rng["x_range"], selector=dict(anchor=xaxis_name))
        fig_resampler.figure.update_yaxes(range=rng["y_range"], selector=dict(anchor=yaxis_name))

    fig_resampler.plot.figure = fig_resampler.figure
    fig_resampler.plot.update()


def show(fig_resampler, options: Optional[dict] = None) -> None:
    """
    Show the figure initially. Sets up the Plotly figure in NiceGUI and attaches event handlers.
    """
    if options is None:
        options = {}
    fig_resampler._resample_all_traces()
    fig_dict = fig_resampler.figure.to_dict()
    fig_dict["config"] = options

    if fig_resampler.plot:
        fig_resampler.plot.figure = fig_resampler.figure
        fig_resampler.plot.update()
    else:
        fig_resampler.plot = ui.plotly(fig_dict)
        fig_resampler.plot.on("plotly_relayout", partial(_on_relayout, fig_resampler))
        fig_resampler.plot.on("plotly_doubleclick", partial(_on_doubleclick, fig_resampler))


def update(fig_resampler) -> None:
    """
    Update the figure after changes have been made (e.g., after reset, adding new traces).
    Re-run the resampling and refresh the displayed figure.
    """
    fig_resampler._resample_all_traces()
    fig_resampler.plot.figure = fig_resampler.figure
    fig_resampler.plot.update()
