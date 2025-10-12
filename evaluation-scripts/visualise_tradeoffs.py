#!/usr/bin/env python3
# plot_tradeoffs_single.py
# Single-classifier trade-off plots:
# grey line + λ-colored markers + red Pareto rings.
# ---------------------------------------------------------------------------

import sys, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, FormatStrFormatter
from matplotlib.colors import SymLogNorm

# -------------------- CLI ----------------------------------------------------
p = argparse.ArgumentParser(description="Trade-off plots for a single classifier")
p.add_argument("all_df",   help="CSV with λ results")
p.add_argument("front_df", help="CSV with Pareto frontier rows (must contain 'pareto')")
p.add_argument("prop",     help="Property/compliance column to plot (e.g., LRO5, COMP, RuleScore)")
p.add_argument("--out",    help="Output file (png/pdf/svg). If omitted, show interactively.")
args = p.parse_args()

prop_col = args.prop
REQ = ['lambda', prop_col, 'KL', 'FCD', 'Validity']

def display_name(name):
    if name == "COMP": return "CONJ"
    if name == "LRO5": return "LR"
    return name

def load_and_check(path, need_pareto=False):
    df = pd.read_csv(path)
    for c in REQ:
        if c not in df.columns:
            sys.exit(f"Column '{c}' missing in {path}")
    if need_pareto and 'pareto' not in df.columns:
        sys.exit(f"'pareto' column missing in {path}")
    return df

# -------------------- Data ---------------------------------------------------
df_all = load_and_check(args.all_df)
df_pf  = load_and_check(args.front_df, need_pareto=True)

# -------------------- λ normalization (symlog) -------------------------------
all_lams = df_all['lambda'].astype(float).values
vmax = all_lams.max()
norm = SymLogNorm(linthresh=1, vmin=0, vmax=vmax, base=10)
cmap = mpl.cm.viridis

# -------------------- Style --------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
})

NEUTRAL = "0.5"
MARKER  = 'o'

# -------------------- Layout -------------------------------------------------
fig = plt.figure(figsize=(12.0, 8.5))
gs  = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.30)

axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[1,1]),
]

specs = [
    (prop_col, 'KL',         f'{display_name(prop_col)} vs KL'),
    (prop_col, 'FCD',        f'{display_name(prop_col)} vs FCD'),
    ('Validity', 'KL',       'Validity vs KL'),
    (prop_col, 'Validity',   f'{display_name(prop_col)} vs Validity'),
]

# -------------------- Plot function ------------------------------------------
def plot_classifier(ax, df, pf_df, marker, want_sc_ref=False):
    df_sorted = df.sort_values('lambda')
    colors    = df_sorted['lambda'].astype(float).values
    pf        = pf_df[pf_df['pareto']].copy()
    pf_colors = pf['lambda'].astype(float).values

    # thin neutral line
    ax.plot(df_sorted[xcol], df_sorted[ycol],
            color=NEUTRAL, linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)

    # λ-colored points
    sc = ax.scatter(df_sorted[xcol], df_sorted[ycol],
                    c=colors, cmap=cmap, norm=norm,
                    s=80, marker=marker, edgecolor="0.3", linewidth=0.6,
                    alpha=0.95, zorder=2)

    # Pareto overlay
    ax.scatter(pf[xcol], pf[ycol],
               c=pf_colors, cmap=cmap, norm=norm,
               s=90, marker=marker, facecolors='none',
               edgecolor='red', linewidth=1.2, zorder=3)

    return sc if want_sc_ref else None

# -------------------- Plot all panels ----------------------------------------
sc_ref = None
for ax, (xcol, ycol, title) in zip(axes, specs):
    sc_ref = plot_classifier(ax, df_all, df_pf, MARKER, want_sc_ref=True)

    ax.set_xlabel(display_name(xcol))
    ax.set_ylabel(display_name(ycol))
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(True, which='major', alpha=0.65, linewidth=0.9, linestyle='--')
    ax.grid(True, which='minor', alpha=0.35, linewidth=0.7, linestyle=':')

# -------------------- Panel letters ------------------------------------------
for letter, ax in zip("abcd", axes):
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes,
            ha='left', va='bottom', fontsize=13, weight='bold')

# -------------------- Colorbar -----------------------------------------------
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=axes, shrink=0.95, pad=0.02)
cbar.set_label("λ (symlog scale)")
tick_vals = [0, 1, 3, 10, 30, 100, 300, int(vmax)]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([str(v) for v in tick_vals])

# -------------------- Save or show ------------------------------------------
if args.out:
    fig.savefig(args.out, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
else:
    plt.show()

