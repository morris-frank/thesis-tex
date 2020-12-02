#!/usr/bin/env python

import os
from functools import partial
from itertools import chain, product

import ipdb
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
    with open("preamble/colors.def", "w") as fp:
        for name, hex in chain(CMAP_CAT.items(), COLORS.items()):
            rgb = str(hex2rgb(hex))[1:-1]
            fp.write(f"\\definecolor{{{name}}}{{rgb}}\t{{{rgb}}}\n")


@log_func()
def plot_waveforms(signals):
    import librosa
    import librosa.display

    for signal in tqdm(signals, leave=False):
        wave, _ = librosa.load(f"data/sounds/{signal}.wav")
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
        plot_heatmap(log_p, name, signals)
        pd.DataFrame(log_p, columns=signals, index=signals).to_latex(
            f"{FIGDIR}/{name}_hm.tex", float_format="%.1e"
        )
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
            # axs[k].set_xlim(-100, 15)
            # axs[k].set_xscale('log')
            # axs[k].legend()
        plt.show()
        ipdb.set_trace()
    else:
        raise ValueError("HAHAHA")


@log_func()
def plot_training_curves(filename, signals, ylim, bs, shapes=None, xlim=None):
    df = pd.read_csv(f"./data/training/{filename}")
    df = df.apply(partial(pd.to_numeric, errors="coerce")).fillna(0)
    for signal in signals:
        cols = [c for c in df.columns if signal in c]
        if shapes is not None:
            for shape in shapes:
                col = [c for c in cols if shape in c][0]
                df[signal + "_" + shape] = df[col]
        else:
            df[signal] = df[cols[0]]
            if len(cols) > 1:
                df[signal] += df[cols[1]]
    df.index.rename("Step", inplace=True)
    cols = signals
    if shapes is not None:
        cols = [s + "_" + sh for s, sh in product(signals, shapes)]
    df = (
        df[cols]
        .rolling(30, min_periods=1)
        .mean()
        .melt(value_name="log(p)", var_name="Source", ignore_index=False)
        .reset_index()
    )

    if shapes is not None:
        df[["Source", "Noise"]] = df.Source.str.split("_", expand=True)
    df["Step"] *= bs

    _, ax = plt.subplots(
        figsize=(BODY_LENGTH, 0.9 * MARGIN_LENGTH),
        gridspec_kw=dict(
            bottom=0.25, top=0.95, right=0.8 if shapes is not None else 0.95
        ),
    )
    sns.lineplot(
        x="Step",
        y="log(p)",
        hue="Source",
        data=df,
        ax=ax,
        linewidth=0.5,
        style="Noise" if shapes is not None else None,
    )
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if shapes is not None:
        plt.legend(bbox_to_anchor=(1.04, 0.4), loc="center left")
    savefig(os.path.basename(filename)[:-4] + "/train")


@log_func()
def plot_const(data, prefix=""):
    data = data[["const_levels", "const_logp"]].flatmap().reset_index()
    data = data[data.const_levels.isin([0.0, 1.0])]
    data = data[data["index"].isin(["0.0", "0.359"])]
    data = data.set_index(["index", "const_levels"])
    data = pd.DataFrame(data.const_logp.tolist(), index=data.index, columns=TOY_SIGNALS)
    data.index.rename(("model", "value"), inplace=True)
    data = data.swaplevel()
    data.sort_index(axis=0, level=0, inplace=True)
    data.to_latex(f"{FIGDIR}/{prefix}toy_const.tex", float_format="%.1e")


@log_func()
def plot_noised_noised(data, prefix=""):
    noised_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    noise_levels = data.index
    data = data["noised"]

    rows = {s: [] for s in TOY_SIGNALS}
    for cond, val in data.items():
        val = val.mean(-1).T
        for _val, signal in zip(val, TOY_SIGNALS):
            rows[signal].append(_val)

    for signal, mat in rows.items():
        for i in range(len(mat)):
            mat[i] = mat[i][: len(noised_levels)]
        df = pd.DataFrame(mat, index=noise_levels, columns=noised_levels).T
        df = df.drop([0.01, 0.2], axis=0)
        plot_heatmap(df, f"noised_noised/{prefix}{signal}", ticks="")
        df.to_latex(f"{FIGDIR}/noised_noised/{prefix}{signal}.tex", float_format="%.1e")


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
def plot_noise(data, prefix=""):
    fig, axs = plt.subplots(
        1,
        1,
        figsize=(BODY_LENGTH, BODY_LENGTH * 5 / 12),
        gridspec_kw=dict(
            left=0.2,
            right=0.72,
            top=0.95,
            bottom=0.25,
        ),
    )

    plot_levels(
        data,
        ("noise_levels", "$\sigma$ Noise input"),
        ("noise_logp", "$log(p)$"),
        "Training\nnoise",
        axs,
        ["0.077"],
    )
    ipdb.set_trace()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    savefig(f"{prefix}const_noise_ll")


@log_func()
def plot_magphase():
    import librosa

    wav, sr = librosa.load("./data/sounds/mix.wav")
    D = librosa.stft(wav)
    ν, φ = librosa.magphase(D)

    _, axs = plt.subplots(
        2,
        1,
        figsize=(MARGIN_LENGTH, MARGIN_LENGTH),
        gridspec_kw=dict(
            left=0.0,
            right=1.0,
            top=1.0,
            bottom=0.0,
            hspace=0.05,
        ),
    )
    for ax, m in zip(axs, [ν, np.angle(φ)]):
        ax.imshow(m[:170], cmap=CMAP_DIV)
        ax.axison = False
    plt.savefig("./graphics/magphase.png", dpi=300)


def main():
    cprint("Will process all data figures:", Fore.CYAN)

    wn_musdb = npzload(f"musdb/Oct*")
    plot_cross_likelihood(
        np.log(wn_musdb["channels"]), "musdb_noiseless/wn_", MUSDB_SIGNALS
    )
    exit()

    # wn_prior_data = {}
    # for level in ["0", "01", "027", "077", "129", "359"]:
    #     wn_prior_data[f"0.{level}"] = npzload(f"toy/*{level}.")
    # wn_prior_data = pd.DataFrame(wn_prior_data).T.sort_index()
    # for col in ['noised', 'channels', 'noise_logp', 'const_logp']:
    #     wn_prior_data[col] = wn_prior_data[col].map(np.log)

    # for level in ["0", "01", "027", "077", "129", "359"]:
    #     name = f"toy_noise_{level}/wn_"
    #     _data = wn_prior_data.T["0." + level]
    #     plot_cross_likelihood(_data.channels, name, TOY_SIGNALS)
    # plot_noised_noised(wn_prior_data, prefix="wn_")
    # plot_const(wn_prior_data, prefix="wn_")
    # plot_noise(wn_prior_data, prefix="wn_")

    musdb_prior = npzload("musdb/Aug10-")
    plot_cross_likelihood(musdb_prior["channels"], "musdb_noiseless/", MUSDB_SIGNALS)

    toy_prior_data = {"0.0": npzload("toy/Jul31-1847")}
    noise_levels = ["01", "027", "077", "129", "359"]
    for level in noise_levels:
        toy_prior_data[f"0.{level}"] = npzload(f"toy/Aug07-1757*{level}")
    toy_prior_data = pd.DataFrame(toy_prior_data).T.sort_index()

    # for level in ["0"] + noise_levels:
    #     name = f"toy_noise_{level}/"
    #     _data = toy_prior_data.T["0." + level]
    #     plot_toy_samples(_data.samples, name)
    #     plot_cross_likelihood(_data.channels, name, TOY_SIGNALS)

    # plot_noised_noised(toy_prior_data)
    # plot_training_curves("toy_noise_conditioned.csv", TOY_SIGNALS, [-3, -5.7], 15, shapes=noise_levels, xlim=[0, 60_000])
    # plot_training_curves("musdb_noiseless.csv", MUSDB_SIGNALS, [-2, -8], 15)
    # plot_training_curves("toy_noise_0.csv", TOY_SIGNALS, [-1, -5.5], 10)
    # plot_const(toy_prior_data)
    plot_noise(toy_prior_data)
    plot_waveforms(MUSDB_SIGNALS + ["mix"])
    plot_prior_dists(MUSDB_SIGNALS)
    plot_magphase()


if __name__ == "__main__":
    main()
