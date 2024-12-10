from niceplotly_resampler import FigureResampler
from nicegui import ui
import numpy as np

def generate_raw_data(total_points=100_000):
    """
    Generate synthetic data for demonstration.
    """
    x = np.linspace(0, 100, total_points)
    y1 = np.sin(x) + np.random.normal(scale=0.1, size=total_points)
    y2 = np.cos(x) + np.random.normal(scale=0.1, size=total_points)
    return x, y1, y2

x, y1, y2 = generate_raw_data()
plotly = FigureResampler(num_points=1000, webgl=True)
plotly.add_trace(x=x, y=y1, name="Trace 1")
plotly.add_trace(x=x, y=y2, name="Trace 2")
plotly.update_layout(showlegend=True)


ui.markdown("### Interactive Resampled Plot")
plotly.show(options={"displayModeBar": False})

ui.run()