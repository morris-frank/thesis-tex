import colorsys
import inspect
import os
import re
from itertools import product
from math import sin, cos
from math import tau as τ
from random import random, randint
from typing import Tuple

import numpy as np
import wandb
from colorama import Fore
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import square, sawtooth
from scipy.stats import multivariate_normal


def cprint(string, color=Fore.YELLOW, end="\n"):
    print(f"{color}{string}{Fore.RESET}", end=end, flush=True)


def adapt_colors(target, dicti):
    _h, _s, _v = colorsys.rgb_to_hsv(*target)
    for k in dicti:
        h, s, v = colorsys.rgb_to_hsv(*hex2rgb(dicti[k]))
        dicti[k] = rgb2hex(*colorsys.hsv_to_rgb(h, _s, _v))


def savefig(name):
    # plt.gca().patch.set_alpha(0.)
    os.makedirs(os.path.dirname(f"./figures/{name}.pdf"), exist_ok=True)
    plt.savefig(
        f"./figures/{name}.pdf",
        transparent=True,
        bbox_inches=0,
        facecolor="none",
        edgecolor="none",
    )


def hex2rgb(hex):
    if hex[0] == "#":
        hex = hex[1:]
    rgb = hex[:2], hex[2:4], hex[4:6]
    return tuple(round(int(c, 16) / 255, 2) for c in rgb)


def rgb2hex(r, g, b):
    return "#" + "".join([f"{int(x*255):x}" for x in (r, g, b)])


def add_plot_tick(
    ax: plt.Axes,
    symbol: str,
    pos: float = 0.5,
    where: str = "x",
    size: float = 0.05,
    linewidth: float = 1,
):

    if "x" in where:
        anchor, loc = (pos, 1.01), 8
    else:
        anchor, loc = (-0.025, pos), 7

    _ax = inset_axes(
        ax,
        width=size,
        height=size,
        bbox_transform=ax.transAxes,
        bbox_to_anchor=anchor,
        loc=loc,
    )
    _ax.axison = False

    x = np.linspace(0, τ)

    if "sin" in symbol:
        y = np.sin(x)
        _ax.plot(x, y, linewidth=linewidth, c="k")
    elif "tri" in symbol:
        y = sawtooth(x, width=0.5)
        _ax.plot(x, y, linewidth=linewidth, c="k")
    elif "saw" in symbol:
        y = sawtooth(x, width=1.0)
        _ax.plot(x, y, linewidth=linewidth, c="k")
    elif "sq" in symbol:
        y = square(x)
        _ax.plot(x, y, linewidth=linewidth, c="k")
    elif symbol in ["drums", "bass", "voice", "other"]:
        icon = plt.imread(f"./figures/musdb/{symbol}.png")
        _ax.imshow(np.repeat(icon[..., None], 3, 2))
    else:
        raise ValueError("unknown symbol")


def make_a_rand_dist(ax, N=None, cmap=None):
    def rand(v):
        return 2 * v * random() + (1 - v)

    gw = 200
    if N is None:
        N = randint(4, 7)
    centroids = np.random.rand(N, 2)

    X, Y = np.mgrid[0 : 1 : 1 / gw, 0 : 1 : 1 / gw]
    pts = np.dstack((X, Y))
    Z = np.zeros(pts.shape[:-1])
    for μ in centroids:
        a = rand(0.2)  # Amplitude
        σx, σy = rand(0.5), rand(0.5)  # Variances
        φ = τ * random()  # Angle
        R = np.array([[cos(φ), -sin(φ)], [sin(φ), cos(φ)]])
        Σ = np.array([[0.02 * σx, 0], [0, 0.02 * σy]])
        Σ = R @ Σ @ R.T
        rv = multivariate_normal(μ, Σ)
        Z += a * 0.3 * rv.pdf(pts)

    # lx, ly, lz = hillclimber(199, 199, Z, gw)
    # lx, ly = lx/gw, ly/gw
    # lu, lv, lw = np.gradient(lx), np.gradient(ly), np.gradient(lz)

    ax.plot_surface(X, Y, Z, cmap=cmap, zorder=1, linewidths=(0.05))
    # ax.quiver(lx, ly, lz + 0.01, lu, lv, lw, zorder=10, normalize=True, length=0.08, arrow_length_ratio=0.3, linewidths=(0.1))

    plt.axis("off")


def hillclimber(px, py, Z, gw):
    pz = Z[px, py]
    pts = [(px, py, pz)]
    while True:
        xy = [
            (np.clip(x + px, 0, gw - 1), np.clip(y + py, 0, gw - 1))
            for x, y in product((-1, 0, 1), (-1, 0, 1))
        ]
        z = [Z[x, y] for x, y in xy]
        x, y, z = *xy[np.argmax(z)], np.max(z)
        if x == px and y == py and z == pz:
            break
        pts.append((px, py, pz))
        px, py, pz = x, y, z

    return map(np.array, zip(*pts))


def oscillator(length: int, shape: str, ν: int, φ: int = 0) -> np.ndarray:
    assert 0 <= φ <= ν
    shape = shape.lower()
    x = np.linspace(0, length + ν - 1, length + ν)
    if shape == "triangle":
        y = sawtooth(τ * x / ν, width=0.5)
    elif shape == "saw":
        y = sawtooth(τ * x / ν, width=1.0)
    elif shape == "reversesaw":
        y = sawtooth(τ * x / ν, width=0.0)
    elif shape == "square":
        y = square(τ * x / ν)
    elif shape == "halfsin":
        _y = np.zeros(ν)
        _y[: ν // 2] = np.sin(τ * np.linspace(0, 1, ν // 2))
        y = np.tile(_y, x.shape[0] // ν + 1)
    elif shape == "noise":
        y = np.random.rand(*x.shape)
        y *= 0.1
    elif shape == "sin":
        y = np.sin(τ * x / ν)
    else:
        raise ValueError("Invalid shape given")
    y = y[φ : φ + length][None, ...]
    return y


def key2freq(n: int) -> float:
    return 440 * 2 ** ((n - 49) / 12)


def rand_period_phase(high: int = 88, low: int = 1, sr: int = 16000) -> Tuple[int, int]:
    key = randint(low, high * 10) / 10
    freq = key2freq(key)
    ν = int(sr // freq)
    φ = randint(0, ν)
    return ν, φ


def clip_noise(wave, amount=1.0):
    return np.clip(wave + (amount * np.random.randn(*wave.shape)), -1, 1)


def flip(x):
    return np.vstack(list(reversed(np.split(x, 2))))


def squeeze(x):
    N, L = x.shape
    return x.reshape(N, L // 2, 2).transpose(0, 2, 1).reshape(N * 2, L // 2)


def get_wandb(run_id):
    api = wandb.Api()
    run = api.run(f"/morris-frank/thesis/runs/{run_id}")
    df = run.history()
    return df


def get_func_arguments():
    func_name = inspect.stack()[1].function.strip()
    code_line = inspect.stack()[2].code_context[0].strip()
    try:
        argument_string = re.search(rf"{func_name}\((.*)\)", code_line)[1]
    except TypeError:
        import ipdb

        ipdb.set_trace()
    arguments = re.split(r",\s*(?![^()]*\))", argument_string)
    return arguments


CUR_LOG_LEVEL = 0


def log_func(level=0):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            global CUR_LOG_LEVEL
            indent = "\t" * level
            code_line = inspect.stack()[1].code_context[0].strip()
            mess = f"{indent}- {code_line[:60]}"
            if level != CUR_LOG_LEVEL:
                print("\n")
            cprint(f"{mess}", Fore.WHITE, end="")
            result = func(*args, **kwargs)
            if CUR_LOG_LEVEL > level:
                cprint('\r' + "👍".center(10, '-'), Fore.GREEN)
            else:
                cprint(f"\r{mess}", Fore.WHITE, end="")
                cprint(" 👍", Fore.GREEN)
            CUR_LOG_LEVEL = level
            return result

        return wrapped

    return wrapper
