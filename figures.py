#!/usr/bin/env python
from argparse import ArgumentParser
from math import tau as τ
from itertools import product

from colorama import Fore
from matplotlib import colors, rcParams
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import sawtooth, square
from scipy.stats import multivariate_normal
import librosa
import librosa.display
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# geometry lengths from LaTeX
MARGIN_LENGTH = 2 # Maximum width for a figure in the margin, in inches

# choosen colors
DIVERGING = 'viridis'
DIVERGING = sns.cubehelix_palette(n_colors=6, start=2., rot=0.8, reverse=True, hue=0.65)  # mh is ok, but thats it
PALETTE = {
    'green':        '98971a',
    'yellow':       'd79921',
    'red' :         'cc241d',
    'orange':       'd65d0e',
    'blue':         '458588',
    'purple':       'b16286',
    'aqua':         '689d6a',
    'extlinkcolor': '076678', # external links
    'intlinkcolor': 'af3a03', # internal links
}

# data configs
TOY_SIGNALS = ["sin", "square", "saw", "triangle"]
MUSDB_SIGNALS = ["drums", "bass", "other", "voice"]


def hex2rgb(hex):
    rgb = hex[:2], hex[2:4], hex[4:6]
    return tuple(round(int(c, 16) / 255, 2) for c in rgb)


# mpl config
mpl.style.use("./mpl.style")
mpl.colors._colors_full_map["r"] = hex2rgb(PALETTE['red'])
mpl.colors._colors_full_map["g"] = hex2rgb(PALETTE['green'])
mpl.colors._colors_full_map["b"] = hex2rgb(PALETTE['blue'])
mpl.colors._colors_full_map["c"] = hex2rgb(PALETTE['aqua'])
mpl.colors._colors_full_map["m"] = hex2rgb(PALETTE['purple'])
mpl.colors._colors_full_map["y"] = hex2rgb(PALETTE['yellow'])
mpl.colors._colors_full_map["n"] = hex2rgb(PALETTE['orange'])



def make_a_rand_dist():
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, rv.pdf(pos))
    # ax.contourf(x, y, rv.pdf(pos))


def cprint(string, color = Fore.YELLOW):
    print(f"{color}{string}{Fore.RESET}")


def savefig(name):
    plt.savefig(f"figures/{name}.eps")


def print_color_latex():
    with open("colors.def", "w") as fp:
        for name, hex in PALETTE.items():
            rgb = str(hex2rgb(hex))[1:-1]
            fp.write(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}\n")


def plot_palette():
    def _rplot(ax, colors):
        x = np.linspace(-τ, τ, 200)
        for color in colors:
            y = np.random.rand() * np.sin(x * np.random.rand()) + np.random.rand()
            ax.plot(x, y, c='#' + PALETTE[color])
    fig, axs = plt.subplots(1, 1)

    _rplot(axs, ['red', 'green', 'yellow', 'orange', 'blue', 'purple', 'aqua'])

    plt.show()


def add_plot_tick(ax: plt.Axes, symbol: str, pos: float = 0.5, where: str = "x", size: float = 0.05):

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
        _ax.plot(x, y, linewidth=1, c="k")
    elif "tri" in symbol:
        y = sawtooth(x, width=0.5)
        _ax.plot(x, y, linewidth=1, c="k")
    elif "saw" in symbol:
        y = sawtooth(x, width=1.0)
        _ax.plot(x, y, linewidth=1, c="k")
    elif "sq" in symbol:
        y = square(x)
        _ax.plot(x, y, linewidth=1, c="k")
    elif symbol in ["drums", "bass", "voice", "other"]:
        icon = plt.imread(f"figures/mixing/{symbol}.png")
        _ax.imshow(np.repeat(icon[..., None], 3, 2))
    else:
        raise ValueError("unknown symbol")


def plot_heatmap(data, name, signals):
    n = len(signals)
    fig, ax = plt.subplots(
        1, 1, gridspec_kw=dict(left=0.1, right=1, top=0.86, bottom=0.2), figsize=(MARGIN_LENGTH, 1.15*MARGIN_LENGTH)
    )

    if data.max() - data.min() > 10**2:
        norm = colors.SymLogNorm(linthresh=0.03, base=100)
        ticks = [-100, 0, 7]
    else:
        ticks = None
        norm = None

    cbar_ax = inset_axes(ax, width=1.49, height=0.1, bbox_transform=ax.transAxes, bbox_to_anchor=(0.5, -0.15), loc=8)
    sns.heatmap(data,
        ax=ax,
        annot=False,
        linewidths=2,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "ticks": ticks},
        square=True,
        norm=norm,
        cmap=DIVERGING)

    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 0.9

    for i in range(n):
        add_plot_tick(ax, signals[i], pos=pos_tick[i], where="x", size=size)
        add_plot_tick(ax, signals[i], pos=pos_tick[-i - 1], where="y", size=size)

    savefig(name)


def plot_cross_entropy(name, signals):
    data = np.load(f"data/{name}.npy", allow_pickle=True).item()
    y, logits, logp = np.array(data['y']), np.array(data['ŷ']), np.array(data['logp'])

    data = np.zeros((4, 4))
    for k in range(4):
        data[k,:] = logits[y==k].mean(0)

    plot_heatmap(data, name, signals)


def plot_cross_likelihood(name, signals):
    log_p = np.load(f"data/{name}.npy")
    log_p[log_p == -np.inf] = -1e3
    log_p = np.maximum(log_p, -1e3)
    log_p = log_p.swapaxes(0, 1)
    log_p = log_p.mean(-1)

    plot_heatmap(log_p, name, signals)


def plot_noise_box(name):
    df = np.load(f"data/{name}.npy", allow_pickle=True,).item()

    l = []
    for σ, (i, k) in product(df.keys(), enumerate(TOY_SIGNALS)):
        l.extend([(σ, k, v) for v in df[σ][i].tolist()])

    df = pd.DataFrame(l, columns=['Noise-Level', 'Source', 'Log-Likelihood'])
    df = df[df['Log-Likelihood'] != 0]
    df = df[df['Noise-Level'] != 0.001]
    df = df[df['Noise-Level'] != 0.01]
    df = df[df['Noise-Level'] != 0.05]

    _, axs = plt.subplots(2, 2, figsize=(MARGIN_LENGTH, 1.3*MARGIN_LENGTH), gridspec_kw=dict(left=0.13, right=0.99, hspace=0.5, wspace=0.4))
    for signal, ax in zip(TOY_SIGNALS, axs.flatten()):
        add_plot_tick(ax, symbol=signal, size=0.1)
        sns.boxplot(x="Noise-Level", y="Log-Likelihood", data=df[df['Source'] == signal], ax=ax, fliersize=1, linewidth=0.5, showfliers=False)
        ax.set_ylabel("")
        ax.set_xlabel("")
    savefig(name)


def plot_waveforms():


def main(args):
    if args.verbose:
        cprint("Palette example plot:")
        plot_palette()

    # cprint("Overwriting LaTeX color definitions")
    # print_color_latex()

    # cprint("Will process all data figures:")

    # cprint("– Noise plots likelihood", Fore.GREEN)
    # plot_noise_box('noise_likelihood_with_noise')
    # plot_noise_box('noise_likelihood_without_noise')

    # cprint("– Cross-entropy heatmaps", Fore.GREEN)
    # plot_cross_entropy('heatmap_musdb_classifier', MUSDB_SIGNALS)

    # cprint("– Cross-Likelihood heatmaps", Fore.GREEN)
    # plot_cross_likelihood('heatmap_musdb', MUSDB_SIGNALS)
    # plot_cross_likelihood('heatmap_toy', TOY_SIGNALS)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-v', action='store_true', dest='verbose')
    main(parser.parse_args())
