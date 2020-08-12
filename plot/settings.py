import matplotlib as mpl
import seaborn as sns

from .utils import hex2rgb

FIGDIR = "./graphics"

# geometry lengths from LaTeX
MARGIN_LENGTH = 2  # Maximum width for a figure in the margin, in inches
BODY_LENGTH = 4.21342  # Maximum inner body line widht in inches
FULLWIDTH_LENGTH = 6.5

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
MUSDB_SIGNALS = ["drums", "bass", "other", "vocals"]

# Update matplotlib config
mpl.style.use("./plot/mpl.style")


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
