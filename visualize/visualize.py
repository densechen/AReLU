import numpy as np
from visdom import Visdom
PORT = 8097

_WINDOW_CASH = {}
_ENV_CASH = {}


def _vis(env="main"):
    if env not in _ENV_CASH:
        _ENV_CASH[env] = Visdom(env=env, port=PORT)
    return _ENV_CASH[env]


def visualize_losses(loss: dict, title: str, env="main", epoch=0):
    legend = list()
    scalars = list()
    dash = []
    flag = 0
    for k, v in loss.items():
        legend.append(k)
        scalars.append(v)
        if flag % 3 == 0:
            dash.append("solid")
        elif flag % 3 == 1:
            dash.append("dash")
        elif flag % 3 == 2:
            dash.append("dashdot")
        flag += 1
    dash = np.asarray(dash)
    options = dict(
        width=1200,
        height=600,
        xlabel="Epochs",
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
        legend=legend,
        fillarea=False,
        dash=dash,
    )
    if title in _WINDOW_CASH:
        _vis(env).line(Y=[scalars], X=[epoch],
                       win=_WINDOW_CASH[title], update="append", opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(
            Y=[scalars], X=[epoch], opts=options)


def visualize_accuracy(loss: dict, title: str, env="main", epoch=0):
    legend = list()
    scalars = list()
    dash = []
    flag = 0
    for k, v in loss.items():
        legend.append(k)
        scalars.append(v)
        if flag % 3 == 0:
            dash.append("solid")
        elif flag % 3 == 1:
            dash.append("dash")
        elif flag % 3 == 2:
            dash.append("dashdot")
        flag += 1
    dash = np.asarray(dash)
    options = dict(
        width=1200,
        height=600,
        xlabel="Epochs",
        ylabel="%",
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
        legend=legend,
        fillarea=False,
        dash=dash,
    )
    if title in _WINDOW_CASH:
        _vis(env).line(Y=[scalars], X=[epoch],
                       win=_WINDOW_CASH[title], update="append", opts=options)
    else:
        _WINDOW_CASH[title] = _vis(env).line(
            Y=[scalars], X=[epoch], opts=options)
