"""Shared matplotlib style for all figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 11,
})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
