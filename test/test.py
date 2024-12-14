import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from niceplotly_resampler import FigureResampler
from nicegui import ui
import numpy as np
from plotly_resampler import FigureResampler as FigureResamplerOld
from plotly import graph_objects as go
from plotly.subplots import make_subplots

def generate_raw_data(total_points=100_000):
    """
    Generate synthetic data for demonstration.
    """
    x = np.linspace(0, 100, total_points)
    y1 = np.sin(x) + np.random.normal(scale=0.1, size=total_points)
    y2 = np.cos(x) + np.random.normal(scale=0.1, size=total_points)
    return x, y1, y2

x, y1, y2 = generate_raw_data()
fig = FigureResampler(go.Figure())
fig.add_trace(go.Scattergl(x=x, y=y1, name="Trace 1"))
fig.add_trace(go.Scattergl(x=x, y=y2, name="Trace 2"))
fig.update_layout(showlegend=True)

fig2 = FigureResamplerOld(go.Figure())
fig2.add_trace(go.Scattergl(x=x, y=y1, name="Trace 1"))
fig2.add_trace(go.Scattergl(x=x, y=y2, name="Trace 2"))
fig2.update_layout(showlegend=True)



ui.markdown("### Interactive Resampled Plot")
plot = fig.show(options={"displayModeBar": False})
#fig2.show_dash()
ui.button("Reset", on_click=lambda: load_figure())

def load_figure():
    fig = FigureResampler(make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True))
    fig.add_trace(go.Scattergl(x=x, y=y1, name="Trace 1"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=x, y=y2, name="Trace 2"), row=2, col=2)
    fig.update_layout(showlegend=True)
    plot = fig.show(options={"displayModeBar": False})

ui.run()

