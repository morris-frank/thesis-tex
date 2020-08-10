import colorsys
import inspect
import re
from itertools import product
from math import tau as τ
from random import randint
from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from colorama import Fore
from scipy.signal import square, sawtooth


def cprint(string, color=Fore.YELLOW, end="\n"):
    print(f"{color}{string}{Fore.RESET}", end=end, flush=True)


def adapt_colors(target, dicti):
    _h, _s, _v = colorsys.rgb_to_hsv(*target)
    for k in dicti:
        h, s, v = colorsys.rgb_to_hsv(*hex2rgb(dicti[k]))
        dicti[k] = rgb2hex(*colorsys.hsv_to_rgb(h, _s, _v))


def hex2rgb(hex):
    if hex[0] == "#":
        hex = hex[1:]
    rgb = hex[:2], hex[2:4], hex[4:6]
    return tuple(round(int(c, 16) / 255, 2) for c in rgb)


def rgb2hex(r, g, b):
    return "#" + "".join([f"{int(x*255):x}" for x in (r, g, b)])


def hillclimber(px, py, Z, gw):
    pz = Z[px, py]
    pts = [(px, py, pz)]
    while True:
        xy = [
            (np.clip(x + px, 0, gw - 1), np.clip(y + py, 0, gw - 1))
            for x, y in product((-1, 0, 1), (-1, 0, 1))
        ]
        z = [Z[x, y] for x, y in xy]
        x, y, z = *xy[np.argmax(z)], np.max(z)
        if x == px and y == py and z == pz:
            break
        pts.append((px, py, pz))
        px, py, pz = x, y, z

    return map(np.array, zip(*pts))


def oscillator(length: int, shape: str, ν: int, φ: int = 0) -> np.ndarray:
    assert 0 <= φ <= ν
    shape = shape.lower()
    x = np.linspace(0, length + ν - 1, length + ν)
    if shape == "triangle":
        y = sawtooth(τ * x / ν, width=0.5)
    elif shape == "saw":
        y = sawtooth(τ * x / ν, width=1.0)
    elif shape == "reversesaw":
        y = sawtooth(τ * x / ν, width=0.0)
    elif shape == "square":
        y = square(τ * x / ν)
    elif shape == "halfsin":
        _y = np.zeros(ν)
        _y[: ν // 2] = np.sin(τ * np.linspace(0, 1, ν // 2))
        y = np.tile(_y, x.shape[0] // ν + 1)
    elif shape == "noise":
        y = np.random.rand(*x.shape)
        y *= 0.1
    elif shape == "sin":
        y = np.sin(τ * x / ν)
    else:
        raise ValueError("Invalid shape given")
    y = y[φ : φ + length][None, ...]
    return y


def key2freq(n: int) -> float:
    return 440 * 2 ** ((n - 49) / 12)


def rand_period_phase(high: int = 88, low: int = 1, sr: int = 16000) -> Tuple[int, int]:
    key = randint(low, high * 10) / 10
    freq = key2freq(key)
    ν = int(sr // freq)
    φ = randint(0, ν)
    return ν, φ


def clip_noise(wave, amount=1.0):
    return np.clip(wave + (amount * np.random.randn(*wave.shape)), -1, 1)


def flip(x):
    return np.vstack(list(reversed(np.split(x, 2))))


def squeeze(x):
    N, L = x.shape
    return x.reshape(N, L // 2, 2).transpose(0, 2, 1).reshape(N * 2, L // 2)


def get_wandb(run_id):
    api = wandb.Api()
    run = api.run(f"/morris-frank/thesis/runs/{run_id}")
    df = run.history()
    return df


def get_func_arguments():
    func_name = inspect.stack()[1].function.strip()
    code_line = inspect.stack()[2].code_context[0].strip()
    try:
        argument_string = re.search(rf"{func_name}\((.*)\)", code_line)[1]
    except TypeError:
        import ipdb

        ipdb.set_trace()
    arguments = re.split(r",\s*(?![^()]*\))", argument_string)
    return arguments


CUR_LOG_LEVEL = 0


def log_func(level=0):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            global CUR_LOG_LEVEL
            indent = "\t" * level
            code_line = inspect.stack()[1].code_context[0].strip()
            mess = f"{indent}- {code_line[:60]}"
            if level != CUR_LOG_LEVEL:
                print("\n")
            cprint(f"{mess}", Fore.WHITE, end="")
            result = func(*args, **kwargs)
            if CUR_LOG_LEVEL > level:
                cprint("\r" + "👍".center(10, "-"), Fore.GREEN)
            else:
                cprint(f"\r{mess}", Fore.WHITE, end="")
                cprint(" 👍", Fore.GREEN)
            CUR_LOG_LEVEL = level
            return result

        return wrapped

    return wrapper


def flatmap(self):
    rows = []
    idxs = []
    for idx, row in self.iterrows():
        for _row in zip(*row.values):
            rows.append(_row)
            idxs.append(idx)
        # multrows = func(row)
        # rows.extend(multrows)
    return pd.DataFrame.from_records(rows, index=idxs, columns=self.columns)


pd.DataFrame.flatmap = flatmap
