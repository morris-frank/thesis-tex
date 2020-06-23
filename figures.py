#!/usr/bin/env python
from argparse import ArgumentParser
from math import tau as τ

from colorama import Fore
import matplotlib as mpl
from matplotlib import colors, rcParams
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import font_manager as fm
from scipy.signal import sawtooth, square
import numpy as np
import seaborn as sns

# geometry lengths from LaTeX
MARGIN_LENGTH = 2

# choosen colors
DIVERGING = 'viridis'
PALETTE = {
    'green':        '98971a',
    'yellow':       'd79921',
    'red' :         'cc241d',
    'orange':       'd65d0e',
    'blue':         '458588',
    'purple':       'b16286',
    'aqua':         '689d6a',
    'extlinkcolor': 'ff0000', # external links
    'intlinkcolor': '00ff00', # internal links
}

# data configs
TOY_SIGNALS = ["sin", "square", "saw", "triangle"]
MUSDB_SIGNALS = ["drums", "bass", "other", "voice"]

# mpl config
mpl.style.use("./mpl.style")



def cprint(string, color = Fore.YELLOW):
    print(f"{color}{string}{Fore.RESET}")


def hex2rgb(hex):
    rgb = hex[:2], hex[2:4], hex[4:6]
    return tuple(round(int(c, 16) / 255, 2) for c in rgb)


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


def add_plot_tick(
    ax: plt.Axes, symbol: str, pos: float = 0.5, where: str = "x", size: float = 0.05
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


def plot_cross_likelihood(name, signals):
    log_p = np.load(f"data/{name}.npy")
    log_p[log_p == -np.inf] = -1e3
    log_p = np.maximum(log_p, -1e3)
    log_p = log_p.swapaxes(0, 1)
    log_p = log_p.mean(-1)
    n = len(signals)

    fig, ax = plt.subplots(
        1, 1, gridspec_kw=dict(left=0.23, right=1, top=0.85, bottom=-0.09, wspace=0.2), figsize=(MARGIN_LENGTH, MARGIN_LENGTH), dpi=300
    )

    sns.heatmap(log_p, ax=ax, annot=False, linewidths=2, cbar=False, square=True, norm=colors.SymLogNorm(linthresh=0.03, base=np.e), cmap=DIVERGING)

    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 1.2

    for i in range(n):
        add_plot_tick(ax, signals[i], pos=pos_tick[i], where="x", size=size)
        add_plot_tick(ax, signals[i], pos=pos_tick[-i - 1], where="y", size=size)

    plt.savefig(f"figures/{name}.png")


def main(args):
    if args.verbose:
        cprint("Palette example plot:")
        plot_palette()

    cprint("Overwriting LaTeX color definitions")
    print_color_latex()

    cprint("Will process all data figures:")

    cprint("– Cross-Likelihood heatmaps", Fore.GREEN)
    plot_cross_likelihood('heatmap_musdb', MUSDB_SIGNALS)
    plot_cross_likelihood('heatmap_toy', TOY_SIGNALS)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-v', action='store_true', dest='verbose')
    main(parser.parse_args())
