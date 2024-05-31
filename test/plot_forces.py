import plotly.graph_objects as go
import json
import numpy as np


def plot_forces(data):
    fx, fy, fz = [], [], []
    for i, force in enumerate(data['data']):
        fx.append(force['forces'][0])
        fy.append(force['forces'][1])
        fz.append(force['forces'][2])

    frames = [i for i in range(1, len(data['data']) + 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=fx, mode='lines', name=f'Fx', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=frames, y=fy, mode='lines', name=f'Fy', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=frames, y=fz, mode='lines', name=f'Fz', line=dict(color='blue')))

    fig.update_layout(
        #title=dict(text='Forces', font=dict(size=24, family='Arial', color='black', weight='bold')),
        xaxis_title=dict(text='Frame', font=dict(size=24, family='Arial', color='black', weight='bold')),
        yaxis_title=dict(text='Force [N]', font=dict(size=24, family='Arial', color='black', weight='bold')),
        plot_bgcolor='white',  # White background
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',  # Light grey grid lines
            tickfont=dict(size=20, family='Arial', color='black', weight='bold'),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',  # Light grey grid lines
            zeroline=True,
            zerolinecolor='grey',  # Color for the zero line
            tickfont=dict(size=20, family='Arial', color='black', weight='bold'),
        ),
        legend=dict(
            font=dict(size=24, family='Arial', color='black', weight='bold')  # Enlarge and bold the legend
        ),
        width = 1500,
        height = 1000
    )

    fig.write_image("forces_plot.png")  # Save as SVG file
    fig.show()


if __name__ == "__main__":
    json_path = "../data/Dice/Dice_T1/robot_data.json"
    with open(json_path, 'r') as file:
        data = json.load(file)

    plot_forces(data)