#!/usr/bin/env python
from argparse import ArgumentParser
from colorama import Fore
import numpy as np
from math import tau as τ
from matplotlib import pyplot as plt


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


def example_palette_plot():
    def _rplot(ax, colors):
        x = np.linspace(-τ, τ, 200)
        for color in colors:
            y = np.random.rand() * np.sin(x * np.random.rand()) + np.random.rand()
            ax.plot(x, y, c='#' + PALETTE[color])
    fig, axs = plt.subplots(1, 1)

    _rplot(axs, ['red', 'green', 'yellow', 'orange', 'blue', 'purple', 'aqua'])

    plt.show()


def main(args):
    cprint("Palette example plot:")
    example_palette_plot()

    input('?')

    cprint("Overwriting LaTeX color definitions")
    print_color_latex()

    cprint("Will process all data figures:")


if __name__ == "__main__":
    parser = ArgumentParser()
    main(parser.parse_args())
