# NicePlotly-Resampler

[![PyPI](https://img.shields.io/pypi/v/niceplotly-resampler?color=dark-green)](https://pypi.org/project/niceplotly-resampler/)
[![PyPI downloads](https://img.shields.io/pypi/dm/niceplotly-resampler?color=dark-green)](https://pypi.org/project/niceplotly-resampler/)
[![GitHub license](https://img.shields.io/github/license/Vidpic/niceplotly-resampler?color=orange)](https://github.com/Vidpic/niceplotly-resampler/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Vidpic/niceplotly-resampler?color=blue)](https://github.com/Vidpic/niceplotly-resampler/issues)

`NicePloty-Resampler` is a tool for integrating Plotly into NiceGUI, designed to handle large time series datasets efficiently. To achieve this, it dynamically downsamples data based on the current graph view, ensuring smooth user interactions like panning and zooming.

Instead of implementing its own downsampling algorithms, NicePloty-Resampler utilizes the advanced functionality of [tsdownsample](https://github.com/predict-idlab/tsdownsample), which offers highly optimized methods for selecting representative data points. This allows the library to provide exceptional performance while maintaining simplicity.

By default, NicePloty-Resampler uses the MinMaxLTTB method to reduce datasets to 1000 key points for visualization.

![example](https://github.com/Vidpic/niceplotly-resampler/blob/main/docs/example.gif)

## Install
```bash
pip install niceplotly-resampler
```

## Usage

```python
from nicegui import ui
import plotly.graph_objects as go; import numpy as np
from niceplotly_resampler import FigureResampler

x = np.arange(1_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000

fig = FigureResampler(go.Figure())
fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True, x=x, y=noisy_sin))
fig.update_layout(title='Noisy sine wave example', template='plotly_dark', title_x=0.5)

with ui.row().classes('w-full h-full'):
    fig.show(options={"displayModeBar": False}).classes('w-full h-full')
ui.run()
```
