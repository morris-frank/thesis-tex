#!/usr/bin/env python

import os
from functools import partial
from itertools import chain

import ipdb
import librosa
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from matplotlib import pyplot as plt
from tqdm import tqdm

from plot.plot import savefig, plot_heatmap, plot_signals, make_a_rand_dist
from plot.settings import (
    TOY_SIGNALS,
    MUSDB_SIGNALS,
    MARGIN_LENGTH,
    BODY_LENGTH,
    CMAP_CAT,
    CMAP_DIV,
    COLORS,
    FIGDIR,
)
from plot.utils import hex2rgb, cprint, log_func, npzload


@log_func()
def print_color_latex():
    with open("colors.def", "w") as fp:
        for name, hex in chain(CMAP_CAT.items(), COLORS.items()):
            rgb = str(hex2rgb(hex))[1:-1]
            fp.write(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}\n")


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


@log_func()
def plot_toy_samples(data, name):
    for i, sample in enumerate(data):
        sample = sample[:, None, 500:1000]
        sample[1, ...] = sample[1, ...].clip(-0.15, 0.15) / 0.15
        for j in range(4):
            sample[j, ...] -= sample[j, ...].mean()
        plot_signals(sample, legend=False, height=0.4 * MARGIN_LENGTH, x_labels=False)
        savefig(name + f"{i}_samples")


@log_func()
def plot_cross_likelihood(log_p, name, signals, how="heatmap"):
    name += "channels"
    log_p[log_p == -np.inf] = -1e3
    log_p = np.maximum(log_p, -1e3)
    log_p = log_p.swapaxes(0, 1)
    N = len(signals)

    if how == "heatmap":
        log_p = log_p.mean(-1)
        minimum = -10 if "musdb" in name else "auto"
        plot_heatmap(log_p, name, signals, minimum=minimum)
    elif how == "histogram":
        fig, axs = plt.subplots(N, figsize=(BODY_LENGTH, MARGIN_LENGTH))
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


@log_func()
def plot_training_curves(filename, signals, ylim, bs):
    df = pd.read_csv(filename)
    df = df.apply(partial(pd.to_numeric, errors="coerce")).fillna(0)
    for signal in signals:
        cols = [c for c in df.columns if signal in c]
        df[signal] = df[cols[0]]
        if len(cols) > 0:
            df[signal] += df[cols[1]]
    df.index.rename("Step", inplace=True)
    df = (
        df[signals]
        .rolling(30, min_periods=1)
        .mean()
        .melt(value_name="log(p)", var_name="Source", ignore_index=False)
        .reset_index()
    )
    df["Step"] *= bs

    _, ax = plt.subplots(
        figsize=(BODY_LENGTH, 0.9 * MARGIN_LENGTH), gridspec_kw=dict(bottom=.25,top=.95),
    )
    sns.lineplot(x="Step", y="log(p)", hue="Source", data=df, ax=ax, linewidth=0.5)
    ax.set_ylim(ylim)
    savefig(os.path.basename(filename)[:-4])


@log_func()
def plot_const(data):
    data = data[["const_levels", "const_logp"]].flatmap().reset_index()
    data = data[data.const_levels.isin([0.0, 1.0])]
    data = data[data["index"].isin(["0.0", "0.359"])]
    data = data.set_index(["index", "const_levels"])
    data = pd.DataFrame(data.const_logp.tolist(), index=data.index, columns=TOY_SIGNALS)
    data.index.rename(("model", "value"), inplace=True)
    data = data.swaplevel()
    data.sort_index(axis=0, level=0, inplace=True)
    data.to_latex(f"{FIGDIR}/toy_const.tex", float_format="%.1e")


@log_func
def plot_noised_noised(data):
    noised_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    noise_levels = data.index
    data = data["noised"]

    rows = {s: [] for s in TOY_SIGNALS}
    for cond, val in data.items():
        val = val.mean(-1).T
        for _val, signal in zip(val, TOY_SIGNALS):
            rows[signal].append(_val)

    for signal, mat in rows.items():
        df = pd.DataFrame(mat, index=noise_levels, columns=noised_levels).T
        df = df.drop([0.01, 0.2], axis=0)
        plot_heatmap(df, f"noised_noised/{signal}", ticks="")
        df.to_latex(f"{FIGDIR}/noised_noised/{signal}.tex", float_format="%.1e")


def plot_levels(data, x, y, index, ax, exclude=None):
    data = data[[x[0], y[0]]]
    data = data.flatmap().reset_index()
    data.rename(columns={x[0]: x[1], y[0]: y[1], "index": index}, inplace=True)
    data[index] = data[index].astype("category")
    if exclude is not None:
        data = data[~data[index].isin(exclude)]
    data[y[1]] = data[y[1]].map(np.mean)
    sns.lineplot(x=x[1], y=y[1], hue=index, data=data, ax=ax)
    ax.set_yscale("symlog")
    ax.locator_params(axis="y", numticks=5)


@log_func()
def plot_noise(data):
    fig, axs = plt.subplots(
        1,
        1,
        figsize=(BODY_LENGTH, BODY_LENGTH * 5 / 12),
        gridspec_kw=dict(left=0.2, right=0.72, top=0.95, bottom=0.25,),
    )

    plot_levels(
        data,
        ("noise_levels", "$\sigma$ Noise input"),
        ("noise_logp", "$log(p)$"),
        "Training\nnoise",
        axs,
        ['0.077']
    )
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    savefig("const_noise_ll")


def main():
    cprint("Will process all data figures:", Fore.CYAN)
    noise_levels = ["01", "027", "077", "129", "359"]

    plot_training_curves(f"./data/train_musdb_noiseless.csv", MUSDB_SIGNALS, [-2, -8], 15)
    plot_training_curves(f"./data/train_toy_noiseless.csv", TOY_SIGNALS, [-1, -5.5], 10)

    exit()

    musdb_prior = npzload("Aug10-")
    plot_cross_likelihood(musdb_prior["channels"], "musdb_noiseless/", MUSDB_SIGNALS)

    # for level in conds:
    #     plot_prior(f"toy_discr_noise_{level}", os.path.basename(get_newest_file(f"./data/Aug11-*{level}*"))[:-4], TOY_SIGNALS)

    toy_prior_data = {"0.0": npzload("Jul31-1847")}
    for level in noise_levels:
        toy_prior_data[f"0.{level}"] = npzload(f"Aug07-1757*{level}")
    toy_prior_data = pd.DataFrame(toy_prior_data).T.sort_index()

    for level in ["0"] + noise_levels:
        name = f"toy_noise_{level}/"
        _data = toy_prior_data.T["0." + level]
        plot_toy_samples(_data.samples, name)
        plot_cross_likelihood(_data.channels, name, TOY_SIGNALS)

    exit()
    plot_const(toy_prior_data)
    plot_noise(toy_prior_data)
    plot_noised_noised(toy_prior_data)
    plot_waveforms(MUSDB_SIGNALS + ["mix"])
    plot_prior_dists(MUSDB_SIGNALS)


if __name__ == "__main__":
    main()
