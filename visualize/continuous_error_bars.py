import random
from collections import namedtuple

import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.plotly as py

with open('visualize/color_map.txt', "r") as f:
    color_map = [line.split(" ")[0] for line in f]

class ContinuousErrorBars(object):
    def __init__(self, dicts: dict):
        """ dicts:
                -> ReLU:
                    -> runtime 1
                    -> runtime 2
                    -> ...
                -> ...
        """
        self.dicts = dicts

    def draw(self, filename, ticksuffix=""):
        # X
        for k, v in self.dicts.items():
            x_len = len(v[0])
            break

        x = list(range(1, x_len + 1))
        x_rev = x[::-1]

        # LOOP Y
        y = {k: [] for k in self.dicts.keys()}
        y_lower = {k: [] for k in self.dicts.keys()}
        y_upper = {k: [] for k in self.dicts.keys()}

        for k, v in self.dicts.items():
            for i in range(x_len):
                d = []
                for t in range(len(v)):
                    d.append(v[t][i])
                y[k].append(np.mean(d))
                y_lower[k].append(np.min(d))
                y_upper[k].append(np.max(d))

        y_lower = {k: v[::-1] for k, v in y_lower.items()}

        # TRACE
        data = []
        # UPPER AND LOWER
        for i, k in enumerate(self.dicts.keys()):
            trace = go.Scatter(
                x=x+x_rev,
                y=y_upper[k] + y_lower[k],
                fill="tozerox",
                fillcolor="rbga({}, 0.2)".format(color_map[i]),
                line=dict(color="rgba(255,255,255,0)"),
                mode="none",
                showlegend=False,
                name=k
            )
            data.append(trace)

            # MEAN
            trace = go.Scatter(
                x=x,
                y=y[k],
                line=dict(color="rgb({})".format(color_map[i])),
                mode="lines",
                name=k
            )
            data.append(trace)

        layout = go.Layout(
            paper_bgcolor="rgb(255, 255, 255)",
            plot_bgcolor="rgb(255, 255, 255)",
            xaxis=dict(
                gridcolor="rgb(229, 229, 229)",
                range=[1, x_len],
                showgrid=True,
                showline=False,
                showticklabels=True,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=False,
                ticktext="Epochs",
            ),
            yaxis=dict(
                gridcolor="rgb(229, 229, 229)",
                showgrid=True,
                showline=False,
                showticklabels=True,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=False,
                ticksuffix=ticksuffix
            ),
        )

        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename=filename)
