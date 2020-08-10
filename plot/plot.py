import os
from math import floor, log10, ceil, sin, cos
from math import tau as τ
from random import randint, random

import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import square, sawtooth
from scipy.stats import multivariate_normal

from .utils import get_func_arguments
from .settings import MARGIN_LENGTH, BODY_LENGTH, CMAP_DIV


def plot_signals(
    *signals,
    sharey: bool = True,
    ylim=None,
    legend=True,
    height: float = MARGIN_LENGTH,
    x_labels=True,
):
    arguments = get_func_arguments()
    colores = ["k", "n", "y", "g", "r"]
    N = max(s.shape[0] for s in signals)
    C = max(s.shape[1] for s in signals)
    if not ylim:
        ylim = (min(map(np.min, signals)), max(map(np.max, signals)))

    fig, axs = plt.subplots(
        C,
        N,
        sharex="all",
        sharey="all" if sharey else "none",
        squeeze=False,
        figsize=(BODY_LENGTH, height),
        gridspec_kw=dict(left=0.05, right=1.0, top=0.95, bottom=0.1),
    )
    for k, (signal, name) in enumerate(zip(signals, arguments)):
        for n in range(signal.shape[0]):
            c = colores[k % len(colores)]
            for i in range(C):
                axs[i, n].plot(signal[n, i, :], f"{c}-", label=name, linewidth=0.5)
                axs[i, n].tick_params(labelbottom=x_labels)
                if sharey:
                    axs[i, n].set_ylim(ylim)
    if legend:
        for ax in axs.flatten().tolist():
            ax.legend()
    return fig


def plot_heatmap(data, name, signals=None, ticks="both", minimum="auto", xlabel=None, title=None):
    if ticks == "both":
        ticks = "xy"
    fig, (ax, cbar_ax) = plt.subplots(
        2,
        gridspec_kw=dict(
            left=0.2,
            right=0.95,
            top=0.86,
            bottom=0.1,
            hspace=0.05,
            height_ratios=(0.9, 0.05),
        ),
        figsize=(MARGIN_LENGTH, 1.15 * MARGIN_LENGTH),
    )

    norm = colors.SymLogNorm(linthresh=0.001, base=10)
    if minimum == "auto":
        _min = data.min().min()
        data_min = np.sign(_min) * 10 ** min(10, floor(log10(np.abs(_min))))
    else:
        data_min = minimum
    data_max = 10 ** max(1, ceil(log10(data.max().max())))
    _ticks = [data_min, 0, data_max]

    sns.heatmap(
        data,
        ax=ax,
        annot=False,
        linewidths=1,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "ticks": _ticks},
        square=True,
        norm=norm,
        cmap=CMAP_DIV,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_label_position('top')

    if title is not None:
        ax.set_title(title)

    ax.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labeltop="x" not in ticks,
        labelleft="y" not in ticks,
    )

    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="default")

    if signals is not None:
        N = len(signals)
        pos_tick = np.linspace(0, 1, 2 * N + 1)[1::2]
        size = 1 / N * 0.9

        for i in range(N):
            if "x" in ticks:
                add_plot_tick(ax, signals[i], pos=pos_tick[i], where="x", size=size)
            if "y" in ticks:
                add_plot_tick(ax, signals[i], pos=pos_tick[-i - 1], where="y", size=size)

    savefig(name + "_hm")


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


def savefig(name, dont_close=False):
    # plt.gca().patch.set_alpha(0.)
    os.makedirs(os.path.dirname(f"./figures/{name}.pdf"), exist_ok=True)
    plt.savefig(
        f"./figures/{name}.pdf",
        transparent=True,
        bbox_inches=0,
        facecolor="none",
        edgecolor="none",
    )
    if not dont_close:
        plt.close()
