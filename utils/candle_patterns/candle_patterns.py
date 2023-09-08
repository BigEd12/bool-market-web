import requests
import json
import pandas as pd
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display

def create_candlestick_chart(df, selected_patterns=[]):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    for pattern in selected_patterns:
        # Filter DataFrame to find rows with the selected pattern
        pattern_rows = df[df[pattern] == 1]

        # Add arrows to the plot at the pattern locations
        fig.add_trace(go.Scatter(
            x=pattern_rows['Date'],
            y=pattern_rows['Low'] - 5,  # Adjust the vertical position of arrows
            mode='markers+text',
            marker=dict(size=10, symbol='triangle-up', color='red'),
            text=pattern,
            textposition="bottom center",
            name=f"{pattern} Pattern"
        ))

    return fig