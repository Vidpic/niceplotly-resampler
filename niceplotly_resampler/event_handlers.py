from typing import Any

async def _on_relayout(resampler, event: Any) -> None:
    """
    Handle relayout events to update the x and y axis ranges.

    Args:
        event: The plotly_relayout event.
    """
    args = event.args
    updated_subplots = {}

    for k, v in args.items():
        if ".range[" not in k:
            continue

        axis_part, range_part = k.split(".range[")
        axis_part = axis_part.strip()
        
        row, col = resampler._axis_to_subplot(axis_part)

        if (row, col) not in updated_subplots:
            updated_subplots[(row, col)] = {"x_range": None, "y_range": None}

        idx_str = range_part.rstrip("]")
        idx = int(idx_str)
        val = float(v)

        is_x_axis = axis_part.startswith("xaxis") or axis_part == "xaxis"

        if is_x_axis:
            curr_range = updated_subplots[(row, col)]["x_range"]
            if curr_range is None:
                curr_range = [None, None]
            curr_range[idx] = val
            updated_subplots[(row, col)]["x_range"] = (
                tuple(curr_range) if None not in curr_range else curr_range
            )
        else:
            curr_range = updated_subplots[(row, col)]["y_range"]
            if curr_range is None:
                curr_range = [None, None]
            curr_range[idx] = val
            updated_subplots[(row, col)]["y_range"] = (
                tuple(curr_range) if None not in curr_range else curr_range
            )

    for (row, col), rng_info in updated_subplots.items():
        x_r = rng_info["x_range"]
        y_r = rng_info["y_range"]

        if x_r and None not in x_r:
            resampler.figure.update_layout(xaxis_range=x_r)
        if y_r and None not in y_r:
            resampler.figure.update_layout(yaxis_range=y_r)

    resampler._resample_all_traces()
    resampler.plot.figure = resampler.figure
    resampler.plot.update()

async def _on_doubleclick(resampler, event: Any) -> None:
    """
    Handle double-click events to reset the figure layout and resample all traces.

    Args:
        event: The plotly_doubleclick event.
    """
    resampler.update_layout(xaxis_range=None, yaxis_range=None)
    resampler._resample_all_traces()
    resampler.plot.figure = resampler.figure
    resampler.plot.update()