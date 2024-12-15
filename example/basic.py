import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

