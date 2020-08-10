#!/usr/bin/env python

from functools import partial
from itertools import chain
from math import tau as τ

import ipdb
import librosa
import librosa.display
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

from plot.plot import savefig, plot_heatmap, plot_signals, make_a_rand_dist
from plot.settings import TOY_SIGNALS, MUSDB_SIGNALS, MARGIN_LENGTH, \
    BODY_LENGTH, CMAP_CAT, CMAP_DIV, COLORS, FIGDIR
from plot.utils import hex2rgb, cprint, log_func, get_wandb


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


def plot_cross_entropy(name, signals):
    N = len(signals)
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
    N = len(signals)

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


def plot_toy_samples(data, name):
    for i, sample in enumerate(data):
        sample = sample[:, None, 500:1000]
        sample[1, ...] = sample[1, ...].clip(-0.15, 0.15) / 0.15
        for j in range(4):
            sample[j, ...] -= sample[j, ...].mean()
        plot_signals(sample, legend=False, height=0.4 * MARGIN_LENGTH, x_labels=False)
        savefig(name + f'{i}_samples')


@log_func()
def plot_prior(name, filename, signals):
    # dict_keys(['noised', 'channels', 'noise_levels', 'noise_logp', 'const_levels', 'const_logp', 'samples'])
    data = dict(np.load(f"./data/{filename}.npz"))
    name += "/"
    plot_toy_samples(data['samples'], name)
    plot_cross_likelihood(data["channels"], name, signals)


def plot_noised_noised(data):
    noised_levels = [.0, .001, .01, .05, .1, .2, .3]
    noise_levels = data.index
    data = data['noised']

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


def plot_levels(data, x, y, index, ax):
    data = data[[x[0], y[0]]]
    data = data.flatmap().reset_index()
    data.rename(columns={x[0]: x[1], y[0]: y[1], 'index': index}, inplace=True)
    data[index] = data[index].astype('category')
    data[y[1]] = data[y[1]].map(np.mean)
    sns.lineplot(x=x[1], y=y[1], hue=index, data=data, ax=ax)
    ax.set_yscale('symlog')
    ax.locator_params(axis='y', numticks=5)


@log_func()
def plot_noise(data):
    fig, axs = plt.subplots(1, 1, figsize=(BODY_LENGTH, BODY_LENGTH*5/12), gridspec_kw=dict(
            left=0.2,
            right=0.72,
            top=0.95,
            bottom=0.25,
        ))

    plot_levels(data, ('noise_levels', '$\sigma$ Noise input'), ('noise_logp', '$log(p)$'), 'Training\nnoise', axs)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    savefig('const_noise_ll')


@log_func()
def plot_const(data):
    data = data[['const_levels', 'const_logp']].flatmap().reset_index()
    data = data[data.const_levels.isin([0., 1.])]
    data = data[data['index'].isin(['0.0', '0.359'])]
    data = data.set_index(['index', 'const_levels'])
    data = pd.DataFrame(data.const_logp.tolist(), index=data.index, columns=TOY_SIGNALS)
    data.index.rename(('model', 'value'), inplace=True)
    data = data.swaplevel()
    data.sort_index(axis=0, level=0, inplace=True)
    data.to_latex(f"{FIGDIR}/toy_const.tex", float_format="%.1e")


def main():
    cprint("Will process all data figures:", Fore.CYAN)
    fn_noiseless = 'Jul31-1847_Flowavenet_toy_time_rand_ampl'
    fn_noised = "Aug07-1757_Flowavenet_toy_time_noise_0-{}_rand_ampl"
    conds = ['01', '027', '077', '129', '359']

    toy_prior_data = {'0.' + n: fn_noised.format(n) for n in conds}
    toy_prior_data['0.0'] = fn_noiseless
    for k, v in toy_prior_data.items():
        toy_prior_data[k] = dict(np.load(f"./data/{v}.npz"))
    toy_prior_data = pd.DataFrame(toy_prior_data).T.sort_index()

    plot_const(toy_prior_data)
    plot_noise(toy_prior_data)
    plot_noised_noised(toy_prior_data)
    plot_prior("toy_noiseless", fn_noiseless, TOY_SIGNALS)
    for level in conds:
        plot_prior(f"toy_noise_{level}", fn_noised.format(level), TOY_SIGNALS)
    plot_waveforms(MUSDB_SIGNALS + ["mix"])
    plot_prior_dists(MUSDB_SIGNALS)


if __name__ == "__main__":
    main()
