#!/usr/bin/env python
import numpy as np
from math import tau as τ
from matplotlib import pyplot as plt
from colorama import Fore


colormap = {
    'black':      '282828',
    'red' :       'cc241d',
    'darkred' :   '9d0006',
    'green':      '98971a',
    'darkgreen':  '79740e',
    'yellow':     'd79921',
    'darkyellow': 'b57614',
    'blue':       '458588',
    'darkblue':   '076678',
    'purple':     'b16286',
    'darkpurple': '8f3f71',
    'aqua':       '689d6a',
    'darkaqua':   '427b58',
    'orange':     'd65d0e',
    'darkorange':  'af3a03'
}


def mess(s):
    print(Fore.YELLOW + s + Fore.RESET)


def hex2rgb(hex):
    rgb = hex[:2], hex[2:4], hex[4:6]
    return tuple(round(int(c, 16) / 255, 2) for c in rgb)


def print_latex():
    for name, hex in colormap.items():
        rgb = str(hex2rgb(hex))[1:-1]
        print(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}")


def _rplot(ax, colors):
    x = np.linspace(-τ, τ, 200)
    for color in colors:
        y = np.random.rand() * np.sin(x * np.random.rand()) + np.random.rand()
        ax.plot(x, y, c='#' + colormap[color])

def plot():
    fig, axs = plt.subplots(2, 2)

    _rplot(axs[0, 0], ['red', 'green', 'yellow', 'orange', 'blue', 'purple', 'aqua'])
    _rplot(axs[0, 1], ['black', 'black', 'black'])
    _rplot(axs[1, 0], ['darkred', 'darkgreen', 'darkyellow', 'darkorange', 'darkblue', 'darkpurple', 'darkaqua'])
    _rplot(axs[1, 1], ['darkred', 'darkgreen', 'darkyellow', 'darkorange', 'darkblue', 'darkpurple', 'darkaqua'])

    plt.show()

def main():
    mess('First let me print the LaTeX definitons:\n')
    print_latex()

    mess('\nNext up lets make an example plot:')
    plot()

if __name__ == "__main__":
    main()
