#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import chain
from math import tau as τ

from scipy import stats
import librosa
import librosa.display
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import colors


from utils import *

# geometry lengths from LaTeX
MARGIN_LENGTH = 2  # Maximum width for a figure in the margin, in inches

# choosen colors
# CMAP_DIV = "viridis"
CMAP_DIV = sns.cubehelix_palette(
    n_colors=12, start=2.4, rot=0.8, reverse=True, hue=0.5, dark=0.3, as_cmap=True
)  # mh is ok, but thats it
CMAP_CAT = {
    "purple": "#b16286",
    "orange": "#d65d0e",
    "blue": "#458588",
    "green": "#98971a",
    "red": "#cc241d",
    "yellow": "#d79921",
    "aqua": "#689d6a",
}
COLORS = {
    "extlinkcolor": "#076678",  # external links
    "intlinkcolor": "#af3a03",  # internal links
}

# data configs
TOY_SIGNALS = ["sin", "square", "saw", "triangle"]
MUSDB_SIGNALS = ["drums", "bass", "other", "voice"]


# Update matplotlib config
mpl.style.use("./mpl.style")
for k, c in [
    ("r", "red"),
    ("g", "green"),
    ("b", "blue"),
    ("c", "aqua"),
    ("m", "purple"),
    ("y", "yellow"),
    ("n", "orange"),
]:
    mpl.colors._colors_full_map[k] = hex2rgb(CMAP_CAT[c])
sns.set_palette(sns.color_palette(list(CMAP_CAT.values())))


def print_color_latex():
    with open("colors.def", "w") as fp:
        for name, hex in chain(CMAP_CAT.items(), COLORS.items()):
            rgb = str(hex2rgb(hex))[1:-1]
            fp.write(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}\n")


def plot_palette():
    def _rplot(ax, colors):
        x = np.linspace(-τ, τ, 200)
        for color in colors:
            y = np.random.rand() * np.sin(x * np.random.rand()) + np.random.rand()
            ax.plot(x, y, c=CMAP_CAT[color])

    fig, axs = plt.subplots(1, 1)

    _rplot(axs, ["red", "green", "yellow", "orange", "blue", "purple", "aqua"])

    plt.show()


def plot_heatmap(data, name, signals):
    n = len(signals)
    fig, ax = plt.subplots(
        1,
        1,
        gridspec_kw=dict(left=0.1, right=1, top=0.86, bottom=0.2),
        figsize=(MARGIN_LENGTH, 1.15 * MARGIN_LENGTH),
    )

    if data.max() - data.min() > 10 ** 2:
        norm = colors.SymLogNorm(linthresh=0.03, base=100)
        ticks = [-100, 0, 7]
    else:
        ticks = None
        norm = None

    cbar_ax = inset_axes(
        ax,
        width=1.49,
        height=0.1,
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0.5, -0.15),
        loc=8,
    )
    sns.heatmap(
        data,
        ax=ax,
        annot=False,
        linewidths=2,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "ticks": ticks},
        square=True,
        norm=norm,
        cmap=CMAP_DIV,
    )

    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    pos_tick = np.linspace(0, 1, 2 * n + 1)[1::2]
    size = 1 / n * 0.9

    for i in range(n):
        add_plot_tick(ax, signals[i], pos=pos_tick[i], where="x", size=size)
        add_plot_tick(ax, signals[i], pos=pos_tick[-i - 1], where="y", size=size)

    savefig(name)


def plot_cross_entropy(name, signals):
    data = np.load(f"data/{name}.npy", allow_pickle=True).item()
    y, logits, logp = np.array(data["y"]), np.array(data["ŷ"]), np.array(data["logp"])

    data = np.zeros((4, 4))
    for k in range(4):
        data[k, :] = logits[y == k].mean(0)

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

    df = pd.DataFrame(l, columns=["Noise-Level", "Source", "Log-Likelihood"])
    df = df[df["Log-Likelihood"] != 0]
    df = df[~df["Noise-Level"].isin((0.001, 0.01, 0.05))]

    _, axs = plt.subplots(
        2,
        2,
        figsize=(MARGIN_LENGTH, 1.3 * MARGIN_LENGTH),
        gridspec_kw=dict(left=0.17, right=0.99, hspace=0.5, wspace=0.5),
    )
    for signal, ax in zip(TOY_SIGNALS, axs.flatten()):
        add_plot_tick(ax, symbol=signal, size=0.1, linewidth=0.5)
        sns.boxplot(
            x="Noise-Level",
            y="Log-Likelihood",
            data=df[df["Source"] == signal],
            ax=ax,
            fliersize=1,
            linewidth=0.5,
            showfliers=False,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
    savefig(name)


def plot_waveforms(signals):
    for signal in signals:
        wave, _ = librosa.load(f"data/{signal}.wav")
        # stft = librosa.stft(wave)
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        librosa.display.waveplot(
            wave, max_points=500, max_sr=50, ax=ax, color=CMAP_DIV.colors[-70]
        )
        plt.axis("off")
        savefig(f"wave_{signal}")
        plt.close()


def plot_prior_dists(signals):
    def _plot(pks=None):
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        make_a_rand_dist(ax, N=pks, cmap=CMAP_DIV)
        ax.view_init(25, 45)

    for k in signals:
        _plot()
        savefig(f"dist_{k}")
        plt.close()

        _plot(2)
        savefig(f"dist_{k}_post")
        plt.close()


def plot_toy_dist(signals):
    _, axs = plt.subplots(
        2,
        2,
        figsize=(MARGIN_LENGTH, 1.3 * MARGIN_LENGTH),
        gridspec_kw=dict(left=0.17, right=0.99, hspace=0.5, wspace=0.5),
    )
    for signal, ax in zip(TOY_SIGNALS, axs.flatten()):
        add_plot_tick(ax, symbol=signal, size=0.1, linewidth=0.5)
        wave = clip_noise(oscillator(15000, signal, 200), 0.02)
        sns.distplot(wave, ax=ax, kde=False, bins=20, hist_kws={'alpha': 1})
        ax.tick_params(bottom=True, left=False, labelbottom=True, labelleft=False)
        # hist, _ = np.histogram(wave, np.linspace(-1, 1, 100 + 1))
        # ax.plot(hist)
    savefig(f"toy_dist")


def plot_beta():
    _, ax = plt.subplots()

    ε = 0.00001
    N = 400

    for a,b in zip([0.5, 1.01, 2, 2, 6], [0.5, 1, 2, 5, 1]):
        x = np.linspace(stats.beta.ppf(ε, a, b), stats.beta.ppf(1-ε, a, b), N)
        ax.plot(x * 2 - 1, stats.beta.pdf(x, a, b))
    ax.set_ylim(0, 3)
    savefig("beta")



def plot_posterior_example():
    x = np.linspace(.5, (.8*τ)+.5, 100)

    _, ax = plt.subplots(figsize=(1, 2))
    plt.plot(x[:49], np.sin(x[:49]), 'k:', markersize=0.2)
    plt.plot(x[52:], np.sin(x[52:]), 'k:', markersize=0.2)

    iax = ax.inset_axes([0.5, 0.43, 0.1, 0.2])

    x = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
    iax.fill_between(stats.norm.pdf(x), x, alpha=0.8)
    iax.axison = False
    plt.show()



def main(args):
    if args.verbose:
        cprint("Palette example plot:")
        plot_palette()

    processes = [
        (print_color_latex, ()),
        (plot_posterior_example, ()),
        # (plot_beta, ()),
        # (plot_toy_dist, (TOY_SIGNALS,)),
        # (plot_prior_dists, (MUSDB_SIGNALS,)),
        # (plot_waveforms, (MUSDB_SIGNALS + ["mix"],)),
        # (plot_cross_entropy, ("heatmap_musdb_classifier", MUSDB_SIGNALS)),
        # (plot_cross_likelihood, ("heatmap_musdb", MUSDB_SIGNALS)),
        # (plot_cross_likelihood, ("heatmap_toy", TOY_SIGNALS)),
        # (plot_noise_box, ('noise_likelihood_with_noise',)),
        # (plot_noise_box, ('noise_likelihood_without_noise',)),
    ]

    cprint("Will process all data figures:", Fore.CYAN)
    for f, arg in processes:
        cprint(f"- {f.__name__}", Fore.YELLOW, end="")
        f(*arg)
        cprint("\t👍", Fore.GREEN)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", action="store_true", dest="verbose")
    main(parser.parse_args())
