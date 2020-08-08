#!/usr/bin/env python

import os
from functools import partial
from itertools import chain
from math import floor, log10, ceil

import ipdb
import librosa
import librosa.display
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy import stats
from tqdm import tqdm

from utils import *

FIGDIR = "./figures"

# geometry lengths from LaTeX
MARGIN_LENGTH = 2  # Maximum width for a figure in the margin, in inches
BODY_LENGTH = 4.21342  # Maximum inner body line widht in inches

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
N = 4


# Update matplotlib config
mpl.style.use("./mpl.style")


def set_palettes():
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


set_palettes()


def print_color_latex():
    with open("colors.def", "w") as fp:
        for name, hex in chain(CMAP_CAT.items(), COLORS.items()):
            rgb = str(hex2rgb(hex))[1:-1]
            fp.write(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}\n")


def plot_palette():
    fig, ax = plt.subplots()

    x = np.linspace(-τ, τ, 200)
    for color in CMAP_CAT.keys():
        y = np.random.rand() * np.sin(x * np.random.rand()) + np.random.rand()
        ax.plot(x, y, c=CMAP_CAT[color])

    savefig("palette")


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


def plot_heatmap(data, name, signals, ticks="both", minimum="auto"):
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

    norm = colors.SymLogNorm(linthresh=0.03, base=10)
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
        linewidths=2,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal", "ticks": _ticks},
        square=True,
        norm=norm,
        cmap=CMAP_DIV,
    )

    ax.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labeltop="x" not in ticks,
        labelleft="y" not in ticks,
    )

    pos_tick = np.linspace(0, 1, 2 * N + 1)[1::2]
    size = 1 / N * 0.9

    for i in range(N):
        if "x" in ticks:
            add_plot_tick(ax, signals[i], pos=pos_tick[i], where="x", size=size)
        if "y" in ticks:
            add_plot_tick(ax, signals[i], pos=pos_tick[-i - 1], where="y", size=size)

    savefig(name + "_hm")


def plot_cross_entropy(name, signals):
    data = np.load(f"data/{name}.npy", allow_pickle=True).item()
    y, logits, logp = np.array(data["y"]), np.array(data["ŷ"]), np.array(data["logp"])

    data = np.zeros((N, N))
    for k in range(N):
        data[k, :] = logits[y == k].mean(0)

    plot_heatmap(data, name, signals)


def plot_cross_likelihood(log_p, name, signals, how="heatmap"):
    name += "channels"
    log_p[log_p == -np.inf] = -1e3
    log_p = np.maximum(log_p, -1e3)
    log_p = log_p.swapaxes(0, 1)

    if how == "heatmap":
        log_p = log_p.mean(-1)
        plot_heatmap(log_p, name, signals)
    elif how == "histogram":
        fig, axs = plt.subplots(N, figsize=(BODY_LENGTH, MARGIN_LENGTH), dpi=80)
        for k in range(N):
            p = [np.squeeze(_log_p) for _log_p in log_p[k]]
            for j, (_p, lab) in enumerate(zip(p, signals)):
                axs[k].hist(
                    _p,
                    label=lab,
                    range=(-100, 15),
                    bins=np.linspace(-100, 15, 100),
                    alpha=0.7,
                )
                # sns.distplot(_p, ax=axs[k, j], label=lab)
            # axs[k].set_xlim(-100, 15)
            # axs[k].set_xscale('log')
            # axs[k].legend()
        plt.show()
        ipdb.set_trace()
    else:
        raise ValueError("HAHAHA")


@log_func(1)
def plot_log_levels(log_p, levels, name, signals, exclude=None, minimum="auto"):
    log_p = log_p.swapaxes(1, 0)
    levels = np.array(levels).round(3)
    if log_p.ndim > 2:
        log_p = log_p.mean(-1)
    df = pd.DataFrame(log_p, columns=levels, index=signals)
    if exclude is not None:
        df = df.drop(exclude, axis=1)
    plot_heatmap(df, name, signals, ticks="y", minimum=minimum)
    df.to_latex(f"{FIGDIR}/{name}.tex", float_format="%.2e")


@log_func()
def plot_waveforms(signals):
    for signal in tqdm(signals, leave=False):
        wave, _ = librosa.load(f"data/{signal}.wav")
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        librosa.display.waveplot(
            wave, max_points=500, max_sr=50, ax=ax, color=CMAP_DIV.colors[-70]
        )
        plt.axis("off")
        savefig(f"wave/{signal}")
        plt.close()


@log_func()
def plot_prior_dists(signals):
    def _plot(pks=None):
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        make_a_rand_dist(ax, N=pks, cmap=CMAP_DIV)
        ax.view_init(25, 45)

    for signal in tqdm(signals, leave=False):
        _plot()
        savefig(f"dist/{signal}")
        plt.close()

        _plot(2)
        savefig(f"dist/{signal}_post")
        plt.close()


def plot_toy_dist(signals):
    _, axs = plt.subplots(
        N // 2,
        N // 2,
        figsize=(MARGIN_LENGTH, 1.3 * MARGIN_LENGTH),
        gridspec_kw=dict(left=0.17, right=0.99, hspace=0.5, wspace=0.5),
    )
    for signal, ax in zip(TOY_SIGNALS, axs.flatten()):
        add_plot_tick(ax, symbol=signal, size=0.1, linewidth=0.5)
        wave = clip_noise(oscillator(15000, signal, 200), 0.02)
        sns.distplot(wave, ax=ax, kde=False, bins=20, hist_kws={"alpha": 1})
        ax.tick_params(bottom=True, left=False, labelbottom=True, labelleft=False)
        # hist, _ = np.histogram(wave, np.linspace(-1, 1, 100 + 1))
        # ax.plot(hist)
    savefig(f"toy_dist")


def plot_posterior_example():
    x = np.linspace(0.5, (0.8 * τ) + 0.5, 100)

    _, ax = plt.subplots(figsize=(1, 2))
    plt.plot(x[:49], np.sin(x[:49]), "k:", markersize=0.2)
    plt.plot(x[52:], np.sin(x[52:]), "k:", markersize=0.2)

    iax = ax.inset_axes([0.5, 0.43, 0.1, 0.2])

    x = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
    iax.fill_between(stats.norm.pdf(x), x, alpha=0.8)
    iax.axison = False
    plt.show()


def plot_toy_training_curves():
    df = get_wandb("guo159rh")

    var = ("log_p/{}/train", "log_p_0/{}/train", "log_p_1/{}/train")
    sign = (-1, 1, 1)
    ylims = ((0, 3.8), (0, 10), (-1.1, -0.9))
    fig, axs = plt.subplots(1, 3, figsize=[BODY_LENGTH, MARGIN_LENGTH])

    for vf, ax, s, ylim in zip(var, axs, sign, ylims):
        log_p = (
            df[[vf.format(k) for k in TOY_SIGNALS]]
            .melt(ignore_index=False)
            .reset_index()
        )
        log_p["variable"] = log_p["variable"].apply(lambda x: x.split("/")[1])
        log_p["value"] = log_p["value"].apply(lambda x: x * s)
        sns.lineplot(
            x="index",
            y="value",
            hue="variable",
            data=log_p,
            ax=ax,
            legend=False,
            linewidth=0.5,
        )
        ax.set_ylim(ylim)

    savefig("toy_training_curves")


def plot_toy_noise_condtioned_training_curves():
    noises = {
        "Aug03-1911_Flowavenet_toy_time_noise_rand_ampl": 0.1,
        "Aug03-1157_Flowavenet_toy_time_noise_rand_ampl": 0.3,
    }
    df = pd.read_csv("./data/train_noise_conditioned_toy.csv")
    df = df.apply(partial(pd.to_numeric, errors="coerce"))
    df = df.rolling(10, min_periods=1).mean()
    del df["Step"]
    df = df.melt(value_name="log(p)", ignore_index=False).reset_index()
    df.rename(columns={"index": "Step"}, inplace=True)
    df["source"] = df.variable.apply(lambda x: x.split(" - ")[1].split("/")[1])
    df["noise"] = df.variable.apply(lambda x: noises[x.split(" - ")[0]])
    del df["variable"]

    _, ax = plt.subplots()

    sns.lineplot(
        x="Step", y="log(p)", hue="source", data=df, style="noise", ax=ax, linewidth=0.5
    )
    # ax.set_yscale('log')
    ax.set_ylim([-1, -6])
    plt.show()
    ipdb.set_trace()


def plot_toy_interpolation():
    data = np.load("data/prior_toy_interpolate.npy")

    data = data[:, :, 1500:2000]

    plot_signals(data, legend=False)

    savefig("toy_interpolate_time")


def plot_toy_samples():
    data = np.load("data/prior_toy_sample.npy")
    data = data[:, None, :]
    data[1, ...] = data[1, ...].clip(-0.15, 0.15) / 0.15
    # ipdb.set_trace()
    plot_signals(data, legend=False, height=0.4 * MARGIN_LENGTH, x_labels=False)
    savefig("toy_samples_time")


@log_func()
def plot_prior(name, filename, signals):
    # dict_keys(['noised', 'channels', 'noise_levels', 'noise_logp', 'const_levels', 'const_logp', 'samples'])
    data = dict(np.load(f"./data/{filename}.npz"))
    name += "/"
    plot_cross_likelihood(data["channels"], name, signals)
    plot_log_levels(data["noise_logp"], data["noise_levels"], name + "noise", signals)
    plot_log_levels(
        data["const_logp"][5:],
        data["const_levels"][5:],
        name + "const",
        signals,
        exclude=[0.2, 0.6],
    )
    noised_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    plot_log_levels(
        data["noised"], noised_levels, name + "noised", signals, exclude=[0.01, 0.2]
    )


def main():
    cprint("Will process all data figures:", Fore.CYAN)

    plot_prior("toy_noiseless", "Jul31-1847_Flowavenet_toy_time_rand_ampl", TOY_SIGNALS)
    for level in ['0-01', '0-027', '0-077', '0-129', '0-359']:
        plot_prior(f"toy_noise_{level}", f"Aug07-1757_Flowavenet_toy_time_noise_{level}_rand_ampl", TOY_SIGNALS)
    # plot_waveforms(MUSDB_SIGNALS + ["mix"])
    # plot_prior_dists(MUSDB_SIGNALS)


if __name__ == "__main__":
    main()
