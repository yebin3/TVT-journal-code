#!/usr/bin/env python3
"""The five final IEEE TVT performance figures. Grayscale, vector + 600 DPI."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent
VALS = BASE / "results" / "figure_values.csv"
REW = BASE / "results" / "training_reward.csv"
PNG, PDF = BASE / "figures" / "png", BASE / "figures" / "pdf"

ARMS = ("fixed", "rule", "mlp", "transformer")
BAR = {"fixed": "Fixed-Weight\nDDQN", "rule": "Heuristic-\nWeight DDQN",
       "mlp": "Learned Weight\n(MLP)",
       "transformer": "Transformer Weight\nDDQN (Proposed)",
       "obub": "Oracle-Based\nUpper Bound\n(OBUB)"}
LINE = {"fixed": "Fixed-Weight DDQN", "rule": "Heuristic-Weight DDQN",
        "mlp": "Learned Weight (MLP)",
        "transformer": "Transformer Weight DDQN (Proposed)"}
FILL = {"fixed": "0.72", "rule": "0.72", "mlp": "0.72",
        "transformer": "0.45", "obub": "white"}
HATCH = {"fixed": "//", "rule": "\\\\", "mlp": "xx",
         "transformer": "..", "obub": "||"}
LS = {"fixed": "-", "rule": (0, (5.5, 2.2)),
      "mlp": (0, (7.0, 2.0, 1.2, 2.0)), "transformer": (0, (1.2, 2.2))}
LW = {"fixed": 1.0, "rule": 1.0, "mlp": 1.2, "transformer": 1.6}
MK = {"fixed": "s", "rule": "^", "mlp": "D", "transformer": "o"}


def style():
    pref = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
            "FreeSerif", "DejaVu Serif"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white", "font.family": "serif",
        "font.serif": [next((p for p in pref if p in have), "serif")],
        "mathtext.fontset": "dejavuserif", "font.size": 8.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 7.0, "axes.edgecolor": "black", "axes.linewidth": 0.7,
        "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
        "hatch.linewidth": 0.5, "savefig.bbox": "standard",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load():
    V = {}
    for r in csv.DictReader(VALS.open(newline="")):
        V[(r["metric"], r["arm"])] = (float(r["mean"]),
                                      float(r["sd_across_training_seeds"]))
    return V


def save(fig, stem, png_only=False):
    PNG.mkdir(parents=True, exist_ok=True)
    p = PNG / f"{stem}.png"
    fig.savefig(p, dpi=600, facecolor="white")
    if not png_only:
        PDF.mkdir(parents=True, exist_ok=True)
        fig.savefig(PDF / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    return p


def barfig(V, metric, arms, ylabel, stem, ylim, yticks, note=None,
           ceiling=None, divider_after=None, fmt="{:.3f}", size=(5.6, 4.0),
           legend_bars=False, ceiling_ha="left", ceiling_x=None, rect=None,
           png_only=False, legend_fontsize=7.0):
    """legend_bars/ceiling_ha/rect/png_only are OPT-IN and default to the
    original behaviour, so Figures 2-4 render exactly as before."""
    style()
    fig = plt.figure(figsize=size)
    ax = fig.add_axes(rect if rect is not None
                      else [0.125, 0.235, 0.855, 0.70 if note is None else 0.635])
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.7)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.35, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3.0, color="black")
    pos = {a: (i + (0.5 if divider_after is not None and i > divider_after else 0))
           for i, a in enumerate(arms)}
    handles = []
    for a in arms:
        m, sd = V[(metric, a)]
        ax.bar(pos[a], m, width=0.62, facecolor=FILL[a], edgecolor="black",
               linewidth=0.8, hatch=HATCH[a], zorder=3)
        # Same facecolor, hatch, edge colour and edge width as the bar itself,
        # so the legend key is a faithful swatch of the bar it names.
        handles.append(Patch(facecolor=FILL[a], edgecolor="black",
                             linewidth=0.8, hatch=HATCH[a],
                             label=BAR[a].replace("-\n", "-").replace("\n", " ")))
        ax.errorbar(pos[a], m, yerr=sd, ecolor="black", elinewidth=0.7,
                    capsize=2.4, capthick=0.7, fmt="none", zorder=4)
        ax.text(pos[a], m + sd + (ylim[1] - ylim[0]) * 0.018, fmt.format(m),
                ha="center", va="bottom", fontsize=7.4, zorder=5)
    if divider_after is not None:
        ax.axvline(divider_after + 0.75, color="0.45", linestyle=(0, (3, 2)),
                   linewidth=0.8, zorder=2)
    if ceiling is not None:
        ax.axhline(ceiling[0], color="black", linestyle=(0, (1.5, 2)),
                   linewidth=0.8, zorder=2)
        # HORIZONTAL POSITION ONLY. The text, font, size, colour, vertical
        # placement and anchoring style are all unchanged; ceiling_x merely
        # slides it along the ceiling line when an upper-left legend needs the
        # corner it would otherwise occupy.
        ax.text(-0.55 if ceiling_x is None else ceiling_x,
                ceiling[0] - (ylim[1] - ylim[0]) * 0.012, ceiling[1],
                fontsize=6.5, ha=ceiling_ha, va="top", color="0.2")
    ax.set_xticks([pos[a] for a in arms])
    # The method names move into the legend; the TICKS stay, only their text
    # is cleared. ax.set_xlabel("Method") below is untouched either way.
    ax.set_xticklabels([] if legend_bars else [BAR[a] for a in arms],
                       fontsize=7.0)
    ax.set_xlim(-0.62, max(pos.values()) + 0.62)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_xlabel("Method", labelpad=2.0)
    ax.set_ylabel(ylabel)
    if legend_bars:
        # Same 7.0 pt the names had as x tick labels, so their font and size
        # are preserved; only their location changes.
        leg = ax.legend(handles=handles, loc="upper left", ncol=1,
                        fontsize=legend_fontsize, framealpha=1.0,
                        edgecolor="0.5",
                        fancybox=False, borderpad=0.35, labelspacing=0.3,
                        handlelength=1.8, handletextpad=0.45)
        leg.get_frame().set_linewidth(0.5)
        leg.set_zorder(6)
    if note:
        fig.text(0.125, 0.990, note, ha="left", va="top", fontsize=6.2,
                 color="0.25", linespacing=1.4)
    return save(fig, stem, png_only=png_only)


def rewardfig():
    style()
    rows = list(csv.DictReader(REW.open(newline="")))
    fig = plt.figure(figsize=(5.6, 4.0))
    ax = fig.add_axes([0.125, 0.125, 0.855, 0.775])
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.7)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.35, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3.0, color="black")
    allv = []
    for a in ARMS:
        sub = sorted((int(r["episode"]), float(r["mean_reward"]))
                     for r in rows if r["arm"] == a)
        xs = [p[0] for p in sub]; ys = [p[1] for p in sub]; allv += ys
        ax.plot(xs, ys, color="black", linestyle=LS[a], linewidth=LW[a],
                marker=MK[a], markersize=3.0, markevery=18,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=0.7, label=LINE[a], zorder=3)
    lo, hi = min(allv), max(allv); sp_ = hi - lo
    ax.set_xlim(1, 161); ax.set_xticks([1, 40, 80, 120, 161])
    ax.set_ylim(lo - sp_ * 0.06, hi + sp_ * 0.34)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Training Reward  (episode $\\Sigma$ cohort $U_{\\mathrm{qos}}$)")
    leg = ax.legend(loc="upper left", framealpha=1.0, edgecolor="0.5",
                    fancybox=False, borderpad=0.35, labelspacing=0.3,
                    handlelength=2.6, handletextpad=0.45)
    leg.get_frame().set_linewidth(0.5)
    return save(fig, "fig5_training_reward")


def main():
    V = load()
    out = []
    out.append(("1 QoS Satisfaction", barfig(
        V, "qos", ARMS + ("obub",), "QoS Satisfaction Score", "fig1_qos",
        (0.0, 1.06), [0, .2, .4, .6, .8, 1.0], divider_after=3,
        ceiling=(1.0, "Theoretical Maximum (QoS = 1.0)"), size=(6.6, 4.0),
        note="Error bars: $\\pm$1 SD across 3 training seeds ($n=3$). OBUB (open bar, "
             "right of the divider) is a hindsight\nempirical envelope over the evaluated "
             "methods and a finite oracle search \u2014 not a deployable policy.")))
    out.append(("2 Throughput", barfig(
        V, "thr", ARMS, "Throughput (Mbps)", "fig2_throughput",
        (0.0, 5.6), [0, 1, 2, 3, 4, 5],
        note="Received throughput at the application socket, summed over Uu and PC5. "
             "Duplicate copies and\nprotocol headers are included; this is not "
             "unique-payload goodput. Error bars: $\\pm$1 SD, $n=3$.")))
    out.append(("3 Latency", barfig(
        V, "delay", ARMS, "End-to-End Latency (ms)", "fig3_latency",
        (0.0, 160.0), [0, 40, 80, 120, 160], fmt="{:.1f}",
        note="Arithmetic mean of the end-to-end delay of the FIRST received copy of each "
             "logical packet.\nNever-received packets are excluded; late arrivals are "
             "included. Error bars: $\\pm$1 SD, $n=3$.")))
    out.append(("4 Timely Delivery Ratio", barfig(
        V, "tdr", ARMS, "Timely Delivery Ratio (%)", "fig4_timely_delivery",
        (0.0, 100.0), [0, 20, 40, 60, 80, 100], fmt="{:.2f}",
        ceiling=None,
        note="Unique logical packets whose first copy arrived within the 250 ms deadline, "
             "over matured\nlogical packets. Late arrivals count as failures. "
             "Error bars: $\\pm$1 SD, $n=3$.")))
    out.append(("5 Training Reward", rewardfig()))
    for name, p in out:
        print(f"  Fig {name:<24} {p.name}")
    return 0


FIG1_NOTE = ("Error bars: $\\pm$1 SD across 3 training seeds ($n=3$). OBUB (open bar, "
             "right of the divider) is a hindsight\nempirical envelope over the evaluated "
             "methods and a finite oracle search \u2014 not a deployable policy.")


def fig1_qos_revised():
    """Figure 1, with ONLY two changes, plus the margin shift they force.

    1. The five method names move out of the x tick labels and into a legend
       INSIDE the axes, upper-left corner, at the same 7.0 pt they had as tick
       labels. ax.set_xlabel("Method") is untouched -- same text, same 9.5 pt
       axes.labelsize, same labelpad 2.0, same centred alignment.
    2. The canvas grows in HEIGHT only: 6.6 x 4.0 in -> 6.6 x 5.6 in. The width
       is byte-identical, so no horizontal proportion changes.

    Forced by (1), and reported rather than hidden: the "Theoretical Maximum
    (QoS = 1.0)" annotation slides right along the ceiling line, from x = -0.55
    to x = 1.78. Measured in its original position it occupies x -0.55..+1.06,
    y 0.9666..0.9873 -- precisely the upper-left corner an upper-left legend
    must use -- so the two cannot share it. At x = 1.78 it spans 1.78..3.39,
    which clears the legend (right edge 1.66), both bars underneath it (MLP
    value label top 0.9244, Transformer 0.8990) and the dashed divider at
    x = 3.75. It stays left-aligned, directly beneath the ceiling line it
    labels, and its text, font, size, colour and vertical placement are
    unchanged. Right-aligning it instead was measured and rejected: the string
    is 1.606 x-units wide but only 1.370 are free right of the divider, so it
    would have cut across the divider by 0.306 x-units.

    The axes rectangle changes from [0.125, 0.235, 0.855, 0.635] to
    [0.125, 0.075, 0.855, 0.825]: the 0.235 bottom margin existed only to hold
    the three-line method names that are now in the legend, so that space is
    returned to the plot. This is the "minimal margin adjustment" the change
    requires.

    Data, statistics, bar geometry, fills, hatches, divider, ceiling line,
    y-label, y-limits, y-ticks, tick styling, grid and the top note are all
    exactly as in fig1_qos.
    """
    return barfig(
        load(), "qos", ARMS + ("obub",), "QoS Satisfaction Score",
        "fig1_qos_revised", (0.0, 1.06), [0, .2, .4, .6, .8, 1.0],
        divider_after=3, ceiling=(1.0, "Theoretical Maximum (QoS = 1.0)"),
        size=(6.6, 5.6), legend_bars=True, ceiling_ha="left", ceiling_x=1.78,
        rect=[0.125, 0.075, 0.855, 0.825], png_only=True, note=FIG1_NOTE)


def fig1_qos_revised2():
    """fig1_qos_revised at a larger overall size with a less vertical shape.

    Same figure as fig1_qos_revised -- same data, bars, hatches, legend in the
    upper-left, "Method" x-label, ceiling line and annotation, y-axis. ONLY the
    canvas size and the margins change:

        fig1_qos          6.6 x 4.0 in  -> 3960 x 2400 px, aspect 1.65 (too wide)
        fig1_qos_revised  6.6 x 5.6 in  -> 3960 x 3360 px, aspect 1.18 (too tall)
        THIS              8.4 x 6.1 in  -> 5040 x 3660 px, aspect 1.38

    Both dimensions grow relative to fig1_qos_revised (+27 % wide, +9 % tall,
    +39 % area), while the aspect moves back from 1.18 towards the middle of
    the two earlier attempts, so the plot reads as balanced rather than
    vertically stretched. The plotting rectangle itself goes from 1.22 to 1.39
    wide-to-tall, which is what removes the elongated look of the bars.

    Margins are retuned for the bigger canvas: [0.105, 0.070, 0.880, 0.870].
    The legend and the ceiling annotation were re-measured at this size and
    still cover nothing -- smallest clearance +0.0488 in y (legend over the
    Fixed-Weight bar's value label).
    """
    return barfig(
        load(), "qos", ARMS + ("obub",), "QoS Satisfaction Score",
        "fig1_qos_revised2", (0.0, 1.06), [0, .2, .4, .6, .8, 1.0],
        divider_after=3, ceiling=(1.0, "Theoretical Maximum (QoS = 1.0)"),
        size=(8.4, 6.1), legend_bars=True, ceiling_ha="left", ceiling_x=1.78,
        rect=[0.105, 0.070, 0.880, 0.870], png_only=True, note=FIG1_NOTE)


def fig1_qos_enhanced():
    """QoS Satisfaction with a larger legend and a 0.50-1.02 y-axis.

    Same figure as fig1_qos_revised2 -- identical data, means, SD error bars,
    bar order, hatches, fills, divider, ceiling line, value labels, x-label
    "Method", legend entries and their order, canvas 8.4 x 6.1 in. ONLY these
    change, and only because they were asked for:

    1. LEGEND TEXT 7.0 pt -> 11.0 pt (+57 %). Position (upper-left, inside the
       axes), entry order and wording are untouched. Measured at 11 pt the
       legend box spans x -0.56..+2.12, y 0.9143..1.0125, clearing the tallest
       thing underneath it (the MLP bar's +1 SD cap at 0.8797 plus its 0.796
       value label) by +0.0155 in y, so nothing is covered or clipped.
    2. Y-AXIS 0.0-1.06 -> 0.50-1.02, ticks every 0.05. THE AXIS NO LONGER
       STARTS AT ZERO: the bars are truncated and their heights are therefore
       not proportional to the values. The lowest measured point is the MLP
       bar's -1 SD cap at 0.7119, which stays 0.2119 above the 0.50 floor, so
       no datum and no error bar is cut off. The top is 1.02 rather than 1.00
       so that the dotted "Theoretical Maximum (QoS = 1.0)" line stays visible
       as its own line instead of merging into the top spine.

    Forced by (1): the ceiling annotation slides from x = 1.78 to x = 2.30.
    The larger legend now reaches x = +2.12, so at 1.78 the two boxes would
    overlap. At 2.30 the annotation spans 2.30..3.53, leaving +0.178 to the
    legend and +0.224 to the dashed divider at x = 3.75. Its text, font, size,
    colour and vertical placement are unchanged.
    """
    return barfig(
        load(), "qos", ARMS + ("obub",), "QoS Satisfaction Score",
        "QoS_Satisfaction_Enhanced", (0.50, 1.02),
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
        divider_after=3, ceiling=(1.0, "Theoretical Maximum (QoS = 1.0)"),
        size=(8.4, 6.1), legend_bars=True, ceiling_ha="left", ceiling_x=2.30,
        rect=[0.105, 0.070, 0.880, 0.870], png_only=True, note=FIG1_NOTE,
        legend_fontsize=11.0)


if __name__ == "__main__":
    # `python plot_final.py` still rebuilds all five originals, unchanged.
    # `python plot_final.py fig1_revised` writes ONLY fig1_qos_revised.png.
    if sys.argv[1:2] == ["fig1_revised"]:
        print(f"  {fig1_qos_revised()}")
        raise SystemExit(0)
    if sys.argv[1:2] == ["fig1_revised2"]:
        print(f"  {fig1_qos_revised2()}")
        raise SystemExit(0)
    if sys.argv[1:2] == ["qos_enhanced"]:
        print(f"  {fig1_qos_enhanced()}")
        raise SystemExit(0)
    raise SystemExit(main())
