import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon, t
from adjustText import adjust_text
from matplotlib.ticker import MultipleLocator

# ============================================================
# 1. LOAD DATA
# ============================================================

merged = pd.read_csv("merged_summary.csv")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def ci_half_widths(row, assay):
    """
    Returns asymmetric CI half-widths (upper_err, lower_err)
    for a given row and assay ('radioligand' or 'fluorescence').
    Missing CI values become 0 (no bar drawn).
    """
    upper_err = row[f"pIC50_CI_upper_{assay}"] - row[f"pIC50_{assay}"]
    lower_err = row[f"pIC50_{assay}"] - row[f"pIC50_CI_lower_{assay}"]
    return upper_err, lower_err


def bland_altman_stats(diff):
    """
    Given a Series of differences, returns:
    mean diff, SD, upper LoA, lower LoA, 95% CI of the mean.
    """
    md  = diff.mean()
    sd  = diff.std()
    sem = diff.sem()
    ci  = t.interval(0.95, df=len(diff) - 1, loc=md, scale=sem)
    return md, sd, md + 1.96 * sd, md - 1.96 * sd, ci

def fmt_p(p):
    """Format p-value — scientific notation if very small."""
    if p == 0.0:
        return "< 1e-300"   # true underflow — effectively zero
    if p < 0.0001:
        return f"{p:.2e}"
    return round(p, 4)

def fmt_ratio(val, low, up):
    if pd.isna(val): return "—"
    def fmt_single(v):
        if pd.isna(v): return "—"
        if v >= 10:    return f"{v:.0f}"
        elif v >= 1:   return f"{v:.1f}"
        else:          return f"{v:.2f}"
    if pd.isna(low) or pd.isna(up):
        return fmt_single(val)
    return f"{fmt_single(val)} ({fmt_single(low)}–{fmt_single(up)})"

# ============================================================
# 3. DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS
# ============================================================
print("\nCorrelations (fluorescence vs radioligand pIC50):")
for transporter in ["DAT", "NET", "SERT"]:
    data = merged[merged["Transporter"] == transporter].dropna(
        subset=["pIC50_radioligand", "pIC50_fluorescence"])
    x = data["pIC50_radioligand"].values
    y = data["pIC50_fluorescence"].values
    
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    
    print(f"\n{transporter} (n = {len(x)})")
    print(f"  Pearson  r  = {r_p:.3f}, p = {p_p:.4g}")
    print(f"  Spearman rho = {r_s:.3f}, p = {p_s:.4g}")

merged["delta_pIC50"] = (
    merged["pIC50_fluorescence"] - merged["pIC50_radioligand"]
)

print("Descriptive statistics for ΔpIC50 (fluorescence − radioligand):")
print(
    merged.groupby("Transporter")["delta_pIC50"]
    .agg(["count", "mean", "std", "median", "min", "max"])
)

for transporter in ["DAT", "NET", "SERT"]:
    delta = merged.loc[
        merged["Transporter"] == transporter, "delta_pIC50"
    ].dropna()
    mean = delta.mean()
    ci   = t.interval(0.95, df=len(delta) - 1, loc=mean, scale=delta.sem())
    t_stat, t_p = ttest_1samp(delta, 0)
    try:
        w_stat, w_p = wilcoxon(delta)
        wilcoxon_str = f"Wilcoxon: W = {w_stat:.3f}, p = {w_p:.4g}"
    except Exception as e:
        wilcoxon_str = f"Wilcoxon: {e}"
    print(f"\n{transporter} (n = {len(delta)})")
    print(f"  Mean ΔpIC50 = {mean:.3f}")
    print(f"  95% CI      = {ci[0]:.3f} to {ci[1]:.3f}")
    print(f"  t-test:     t = {t_stat:.3f}, p = {t_p:.4g}")
    print(f"  {wilcoxon_str}")

# ── Save statistics to CSV ───────────────────────────────────

rows = []
for transporter in ["DAT", "NET", "SERT"]:
    delta = merged.loc[
        merged["Transporter"] == transporter, "delta_pIC50"
    ].dropna()
    data = merged[merged["Transporter"] == transporter].dropna(
        subset=["pIC50_radioligand", "pIC50_fluorescence"])
    x = data["pIC50_radioligand"].values
    y = data["pIC50_fluorescence"].values

    mean = delta.mean()
    ci   = t.interval(0.95, df=len(delta)-1, loc=mean, scale=delta.sem())
    t_stat, t_p = ttest_1samp(delta, 0)
    w_stat, w_p = wilcoxon(delta)
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    rows.append({
        "Transporter":   transporter,
        "n":             len(delta),
        "Mean_delta_pIC50": round(mean, 3),
        "CI_lower":      round(ci[0], 3),
        "CI_upper":      round(ci[1], 3),
        "t_stat":        round(t_stat, 3),
        "t_p":           fmt_p(t_p),
        "W_stat":        round(w_stat, 3),
        "W_p":           fmt_p(w_p),
        "Pearson_r":     round(r_p, 3),
        "Pearson_p":     fmt_p(p_p),
        "Spearman_rho":  round(r_s, 3),
        "Spearman_p":    fmt_p(p_s),
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv("statistics_output.csv", index=False)
print("\nStatistics saved to statistics_output.csv")

# ============================================================
# 4. SHARED STYLE SETTINGS
# ============================================================

ARROW = dict(arrowstyle="-", color="grey", lw=0.5, shrinkA=3, shrinkB=3)
ADJ   = dict(expand=(1.4, 1.6), force_points=(0.6, 0.8),
             force_text=(0.5, 0.6), lim=500)


def style_legend(legend):
    """Apply consistent black-border style to a legend."""
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)

# ============================================================
# 5. FIGURE 1 — COMBINED SCATTER (top) + BLAND-ALTMAN (bottom)
# ============================================================
#
# 2x3 grid: scatter plots in row 1, Bland-Altman in row 2.
# Shared x-axis range (SCATTER_LIM) across both rows for
# direct vertical comparison per transporter.

SCATTER_LIM = (3.4, 9.1)
BA_Y        = (-2.0, 2.0)

# Pre-compute BA stats
ba_data = {}
for transporter in ["DAT", "NET", "SERT"]:
    data = merged[merged["Transporter"] == transporter].copy()
    data["mean_pIC50"] = (data["pIC50_radioligand"] + data["pIC50_fluorescence"]) / 2
    data["difference"] = data["pIC50_fluorescence"] - data["pIC50_radioligand"]
    data = data.dropna(subset=["mean_pIC50", "difference"])
    md, sd, upper, lower, ci = bland_altman_stats(data["difference"])
    ba_data[transporter] = (data, md, upper, lower)

BA_MANUAL_OFFSETS = {
    "DAT": {
        "Amphetamine": (-50, 8),
        "MDPV":      (-20,  5),
        "Alpha-PVP": (-65, -8),
        "MDMA":      ( -60,  0),
        "4-CMC":     (  0,  8),
        "3-MMC":     ( -60, -5),
        "4-MMC":     ( 10, -5),
    },
    "NET": {
        "4-CMC": (-30, 10),
        "3-MMC": (0, 15),
        "4-MMC": (0, 5),
        "Alpha-PVP": (0, -10),
        "Cocaine": (-30, -15),
        "MDPV": (0, -15),
        "MDMA": (-60, 0),
        "4-FA": (-10, 3),
        "4-MMC": (0, 8),
    },
    "SERT": {
        "Fluoxetine": (-60, 0),
        "3-MMC": (-60, 0),
    }
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

transporters = ["DAT", "NET", "SERT"]

# ── Row 1: Scatter plots ──────────────────────────────────────────────────────
for col, transporter in enumerate(transporters):
    ax = axes[0, col]

    data = merged[merged["Transporter"] == transporter].dropna(
        subset=["pIC50_radioligand", "pIC50_fluorescence"])
    x = data["pIC50_radioligand"].values
    y = data["pIC50_fluorescence"].values

    xerr = np.nan_to_num(np.array([
        [ci_half_widths(row, "radioligand")[1] for _, row in data.iterrows()],
        [ci_half_widths(row, "radioligand")[0] for _, row in data.iterrows()]]))
    yerr = np.nan_to_num(np.array([
        [ci_half_widths(row, "fluorescence")[1] for _, row in data.iterrows()],
        [ci_half_widths(row, "fluorescence")[0] for _, row in data.iterrows()]]))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt="none", color="grey", linewidth=0.8, capsize=2, zorder=2)
    ax.scatter(x, y, color="black", zorder=3, s=35)

    ax.plot(SCATTER_LIM, SCATTER_LIM,
            linestyle="--", linewidth=0.8, color="black", label="y = x")
    ax.set_xlim(SCATTER_LIM)
    ax.set_ylim(SCATTER_LIM)
    ax.tick_params(axis="both", labelsize=11)

    r, p = pearsonr(x, y)
    p_str = "p < 0.0001" if p < 0.0001 else f"p = {p:.2g}"
    ax.text(0.05, 0.95, f"r² = {r**2:.3f}\n{p_str}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", linewidth=0.8))

    ax.set_title(transporter, fontsize=14)
    ax.set_xlabel("Radioligand pIC$_{50}$", fontsize=14)
    style_legend(ax.legend(fontsize=11, loc="lower right"))

axes[0, 0].set_ylabel("Fluorescence pIC$_{50}$", fontsize=14)

# ── Row 2: Bland-Altman plots ─────────────────────────────────────────────────
for col, transporter in enumerate(transporters):
    ax = axes[1, col]
    data, md, upper, lower = ba_data[transporter]
    x = data["mean_pIC50"].values
    y = data["difference"].values

    ax.fill_between(SCATTER_LIM, lower, upper,
                    color="lightgrey", alpha=0.5, zorder=1)
    ax.scatter(x, y, color="black", zorder=3)

    overrides = BA_MANUAL_OFFSETS.get(transporter, {})
    texts = []
    for _, row in data.iterrows():
        name = row["Compound"]
        xi, yi = row["mean_pIC50"], row["difference"]
        if name in overrides:
            dx, dy = overrides[name]
            ax.annotate(name, (xi, yi),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=10, color="black",
                        arrowprops=dict(arrowstyle="-", color="grey",
                                        lw=0.5, shrinkA=8, shrinkB=3))
        else:
            texts.append(ax.text(xi, yi, name, fontsize=10, color="black"))

    if texts:
        adjust_text(texts, x=x, y=y, ax=ax, arrowprops=ARROW, **ADJ)

    ax.axhline(md,    linestyle="--", linewidth=0.8, color="black", 
               label=f"Mean = {md:.2f}")
    ax.axhline(upper, linestyle=":",  linewidth=0.8, color="black", 
               label=f"Upper LoA = {upper:.2f}")
    ax.axhline(lower, linestyle=":",  linewidth=0.8, color="black",
               label=f"Lower LoA = {lower:.2f}")
    ax.axhline(0,     linestyle="-",  linewidth=0.4, color="black")

    ax.set_xlim(SCATTER_LIM)
    ax.set_ylim(BA_Y)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(transporter, fontsize=14)
    ax.set_xlabel("Mean pIC$_{50}$", fontsize=14)
    style_legend(ax.legend(fontsize=11, loc="lower left"))

axes[1, 0].set_ylabel("Difference\nFluorescence − Radioligand", fontsize=14)

# ── Row labels (A, B, C / D, E, F) ───────────────────────────────────────────
for col, label in enumerate(["A", "B", "C"]):
    axes[0, col].text(-0.08, 1.02, label, transform=axes[0, col].transAxes,
                      fontsize=14, fontweight="bold", va="bottom")
for col, label in enumerate(["D", "E", "F"]):
    axes[1, col].text(-0.08, 1.02, label, transform=axes[1, col].transAxes,
                      fontsize=14, fontweight="bold", va="bottom")

plt.tight_layout(h_pad=4.0)
plt.savefig("figure1_combined.png", dpi=150, bbox_inches="tight")

# ============================================================
# 6. FIGURE S1 — SCATTER PLOTS, SUPPLEMENTARY (labelled)
# ============================================================
#
# Per-transporter axis limits for better spread.
# Compound name labels with adjustText repelling.
# Some compounds use manual offset to stay in frame/to not overlap.

SUPP_LIMITS = {
    "DAT":  (3.7, 9.2),
    "NET":  (6.0, 8.0),
    "SERT": (3.3, 9.2),
}

# Manual offset overrides (in points) for labels that escape the frame/overlap
# Format: {transporter: {compound: (dx, dy)}}
MANUAL_OFFSETS = {
    "DAT": {"Amphetamine": (0, 10),
            "Cocaine": (0, 10),
            "4-CMC": (-50, 0),
            "3-MMC": (10, 0),
            "4-MMC": (-50, 0),
            "Alpha-PVP": (0, -50),
            "4-FA": (5, -10)},
    "NET": {"Alpha-PVP": (-55, 8)},
    "SERT": {"3-MMC": (-50, 0),
             "MDPV": (-40, 15),
             "MPHP": (15, 5)}
}

fig_s1, axes_s1 = plt.subplots(1, 3, figsize=(16, 6))

for ax, transporter in zip(axes_s1, ["DAT", "NET", "SERT"]):
    data = merged[merged["Transporter"] == transporter].dropna(
        subset=["pIC50_radioligand", "pIC50_fluorescence"])
    x = data["pIC50_radioligand"].values
    y = data["pIC50_fluorescence"].values

    xerr = np.nan_to_num(np.array([
        [ci_half_widths(row, "radioligand")[1] for _, row in data.iterrows()],
        [ci_half_widths(row, "radioligand")[0] for _, row in data.iterrows()]]))
    yerr = np.nan_to_num(np.array([
        [ci_half_widths(row, "fluorescence")[1] for _, row in data.iterrows()],
        [ci_half_widths(row, "fluorescence")[0] for _, row in data.iterrows()]]))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt="none", color="grey", linewidth=0.8, capsize=2, zorder=2)
    ax.scatter(x, y, color="black", zorder=3, s=35)

    overrides = MANUAL_OFFSETS.get(transporter, {})
    texts = []
    for _, row in data.iterrows():
        name = row["Compound"]
        xi, yi = row["pIC50_radioligand"], row["pIC50_fluorescence"]
        if name in overrides:
            dx, dy = overrides[name]
            ax.annotate(name, (xi, yi),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color="black",
                        arrowprops=dict(arrowstyle="-", color="grey",
                                        lw=0.5, shrinkA=2, shrinkB=2))
        else:
            texts.append(ax.text(xi, yi, name, fontsize=8, color="black"))

    if texts:
        adjust_text(texts, x=x, y=y, ax=ax, arrowprops=ARROW, **ADJ)

    lim = SUPP_LIMITS[transporter]
    ax.plot(lim, lim, linestyle="--", linewidth=0.8, color="black", label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # 0.5 tick spacing, one decimal place
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    r, p = pearsonr(x, y)
    p_str = "p < 0.0001" if p < 0.0001 else f"p = {p:.2g}"
    ax.text(0.05, 0.95, f"r² = {r**2:.3f}\n{p_str}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", linewidth=0.8))

    ax.set_title(transporter)
    ax.set_xlabel("Radioligand pIC$_{50}$")
    style_legend(ax.legend(fontsize=11, loc="lower right"))

axes_s1[0].set_ylabel("Fluorescence pI$_{50}$")
fig_s1.suptitle("Fluorescence vs Radioligand pIC$_{50}$",
                fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("scatter_plots_supplementary.png", dpi=150, bbox_inches="tight")

# ============================================================
# 7. COMPUTE DAT/SERT SELECTIVITY RATIOS
# ============================================================
#
# Expressed as IC50(SERT) / IC50(DAT) in linear space.
# >1  = DAT-selective (e.g. Alpha-PVP ~10,000-fold)
# <1  = SERT-selective (e.g. MDMA)
# ~1  = non-selective
#
# CI propagation: ratio CI calculated as:
#   lower = IC50_SERT_lower / IC50_DAT_upper
#   upper = IC50_SERT_upper / IC50_DAT_lower
#
# Log10 versions used for Bland-Altman (proportional agreement)
# and Pearson r² (log scale scatter).
# Fluoxetine excluded — no DAT data.
 
dat_rows  = merged[merged["Transporter"] == "DAT"].set_index("Compound")
sert_rows = merged[merged["Transporter"] == "SERT"].set_index("Compound")
common    = dat_rows.index.intersection(sert_rows.index)
 
ratio_df = pd.DataFrame({
    "ratio_radio":  (sert_rows.loc[common, "IC50_radioligand"]
                     / dat_rows.loc[common,  "IC50_radioligand"]),
    "ratio_fluoro": (sert_rows.loc[common, "IC50_fluorescence"]
                     / dat_rows.loc[common,  "IC50_fluorescence"]),
}).dropna()
 
# CI propagation in linear space:
# lower ratio CI = SERT_lower / DAT_upper  (most conservative combination)
# upper ratio CI = SERT_upper / DAT_lower  (least conservative combination)
ratio_df["ratio_radio_CI_lower"]  = (sert_rows.loc[ratio_df.index, "IC50_CI_lower_radioligand"]
                                     / dat_rows.loc[ratio_df.index,  "IC50_CI_upper_radioligand"])
ratio_df["ratio_radio_CI_upper"]  = (sert_rows.loc[ratio_df.index, "IC50_CI_upper_radioligand"]
                                     / dat_rows.loc[ratio_df.index,  "IC50_CI_lower_radioligand"])
ratio_df["ratio_fluoro_CI_lower"] = (sert_rows.loc[ratio_df.index, "IC50_CI_lower_fluorescence"]
                                     / dat_rows.loc[ratio_df.index,  "IC50_CI_upper_fluorescence"])
ratio_df["ratio_fluoro_CI_upper"] = (sert_rows.loc[ratio_df.index, "IC50_CI_upper_fluorescence"]
                                     / dat_rows.loc[ratio_df.index,  "IC50_CI_lower_fluorescence"])
 
# Log versions for statistics and Bland-Altman
ratio_df["log_radio"]  = np.log10(ratio_df["ratio_radio"])
ratio_df["log_fluoro"] = np.log10(ratio_df["ratio_fluoro"])
 
print("\nDAT/SERT selectivity ratio  [IC50(SERT) / IC50(DAT)]:")
print(ratio_df[["ratio_radio", "ratio_radio_CI_lower", "ratio_radio_CI_upper",
                "ratio_fluoro", "ratio_fluoro_CI_lower", "ratio_fluoro_CI_upper"]].round(1).to_string())
 
print("\nCorrelations (DAT/SERT selectivity ratio, log scale):")
x_log = ratio_df["log_radio"].values
y_log = ratio_df["log_fluoro"].values
r_p, p_p = pearsonr(x_log, y_log)
r_s, p_s = spearmanr(x_log, y_log)
print(f"  n = {len(x_log)}")
print(f"  Pearson  r   = {r_p:.3f}, p = {p_p:.4g}")
print(f"  Spearman rho = {r_s:.3f}, p = {p_s:.4g}")
 
# ── Save ratio table to CSV ───────────────────────────────────────────────────
 
def fmt_ratio(val, low, up):
    """Format ratio with CI as 'value (lower-upper)', handling NaN CIs."""
    if pd.isna(val):
        return "—"
    if pd.isna(low) or pd.isna(up):
        return f"{val:.0f}"
    return f"{val:.0f} ({low:.0f}–{up:.0f})"
 
ratio_table = pd.DataFrame({
    "Compound":                ratio_df.index,
    "DAT/SERT (radioligand)":  [fmt_ratio(ratio_df.loc[c, "ratio_radio"],
                                           ratio_df.loc[c, "ratio_radio_CI_lower"],
                                           ratio_df.loc[c, "ratio_radio_CI_upper"])
                                 for c in ratio_df.index],
    "DAT/SERT (fluorescence)": [fmt_ratio(ratio_df.loc[c, "ratio_fluoro"],
                                           ratio_df.loc[c, "ratio_fluoro_CI_lower"],
                                           ratio_df.loc[c, "ratio_fluoro_CI_upper"])
                                 for c in ratio_df.index],
})

# ── Save ratio statistics to CSV ─────────────────────────────────────────────
ratio_rows = []
for c in ratio_df.index:
    ratio_rows.append({
        "Compound":                c,
        "DAT/SERT (radioligand)":  fmt_ratio(ratio_df.loc[c, "ratio_radio"],
                                             ratio_df.loc[c, "ratio_radio_CI_lower"],
                                             ratio_df.loc[c, "ratio_radio_CI_upper"]),
        "DAT/SERT (fluorescence)": fmt_ratio(ratio_df.loc[c, "ratio_fluoro"],
                                             ratio_df.loc[c, "ratio_fluoro_CI_lower"],
                                             ratio_df.loc[c, "ratio_fluoro_CI_upper"]),
    })

ratio_table = pd.DataFrame(ratio_rows)
ratio_table.to_csv("ratio_table.csv", index=False)
print("\nRatio table saved to ratio_table.csv")
print(ratio_table.to_string(index=False))

# ============================================================
# 8. FIGURE 3 — DAT/SERT RATIO: SCATTER + BLAND–ALTMAN
# ============================================================
#
# Manual label offsets used throughout — adjustText unreliable on log scale.


scatter_offsets = {
    "3-MMC":       (  5,   5),
    "4-CMC":       (  5, -10),
    "4-FA":        (  5,   5),
    "4-MMC":       ( -5, -15),
    "Alpha-PiHP":  (-70,  5),
    "Alpha-PVP":   (-75,   5),
    "Amphetamine": (  5,   5),
    "Cocaine":     (  5, -12),
    "MDMA":        ( -5, -12),
    "MDPV":        (  5,   5),
    "MPHP":        (  15, -12),
}

ba_offsets = {
    "3-MMC":       (  5,   5),
    "4-CMC":       (  5, 0),
    "4-FA":        (  5,   5),
    "4-MMC":       ( -25, -25),
    "Alpha-PiHP":  (  5,   5),
    "Alpha-PVP":   ( -65, -12),
    "Amphetamine": (-65, -25),
    "Cocaine":     ( -45, -10),
    "MDMA":        (  5, -20),
    "MDPV":        (  5,   5),
    "MPHP":        (  5, -12),
}

fig3, (ax_sc, ax_ba) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)


# ── Panel A: scatter, log-scaled axes, linear tick labels ────

x_lin = ratio_df["ratio_radio"].values
y_lin = ratio_df["ratio_fluoro"].values
x_log = ratio_df["log_radio"].values
y_log = ratio_df["log_fluoro"].values

ax_sc.scatter(x_lin, y_lin, color="black", zorder=3, s=40)

for name in ratio_df.index:
    xi = ratio_df.loc[name, "ratio_radio"]
    yi = ratio_df.loc[name, "ratio_fluoro"]
    dx, dy = scatter_offsets.get(name, (5, 4))
    ax_sc.annotate(
        name, (xi, yi),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=7.5, color="black",
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5,
                        shrinkA=2, shrinkB=2)
        if abs(dx) > 10 or abs(dy) > 10 else None)

lim_low, lim_high = 0.03, 30000
ax_sc.plot([lim_low, lim_high], [lim_low, lim_high],
           linestyle="--", linewidth=0.8, color="black", label="y = x")
ax_sc.set_xscale("log"); ax_sc.set_yscale("log")
ax_sc.set_xlim(lim_low, lim_high); ax_sc.set_ylim(lim_low, lim_high)

ticks = [0.1, 1, 10, 100, 1000, 10000]
ax_sc.set_xticks(ticks); ax_sc.set_yticks(ticks)
ax_sc.get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_sc.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:g}"))

ax_sc.axhline(1, linestyle=":", linewidth=0.6, color="grey", zorder=1)
ax_sc.axvline(1, linestyle=":", linewidth=0.6, color="grey", zorder=1)

r3, p3 = pearsonr(x_log, y_log)
p3_str = "p < 0.0001" if p3 < 0.0001 else f"p = {p3:.4g}"
ax_sc.text(0.05, 0.95, f"r² = {r3**2:.3f}\n{p3_str}",
           transform=ax_sc.transAxes, fontsize=11, verticalalignment="top",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="black", linewidth=0.8))

ax_sc.set_xlabel("Radioligand  IC$_{50}$(SERT) / IC$_{50}$(DAT)", fontsize=12)
ax_sc.set_ylabel("Fluorescence  IC$_{50}$(SERT) / IC$_{50}$(DAT)", fontsize=12)
style_legend(ax_sc.legend(fontsize=11, loc="lower right"))


# ── Panel B: Bland-Altman in log space ───────────────────────

diff_log = ratio_df["log_fluoro"] - ratio_df["log_radio"]
mean_log = (ratio_df["log_fluoro"] + ratio_df["log_radio"]) / 2
md3, sd3, upper3, lower3, ci3 = bland_altman_stats(diff_log)

# ── Ratio correlations ───────────────────────────────────────

ratio_stats = pd.DataFrame([{
    "Comparison":    "DAT/SERT ratio (log scale)",
    "n":             len(x_log),
    "Pearson_r":     round(r_p, 3),
    "Pearson_p":     fmt_p(p_p),
    "Spearman_rho":  round(r_s, 3),
    "Spearman_p":    fmt_p(p_s),
    "Mean_log_diff": round(md3, 3),
    "CI_lower":      round(ci3[0], 3),
    "CI_upper":      round(ci3[1], 3),
    "LoA_lower":     round(lower3, 3),
    "LoA_upper":     round(upper3, 3),
}])

ratio_stats.to_csv("statistics_ratio.csv", index=False)
print("Ratio statistics saved to statistics_ratio.csv")

# ── Per-compound ratio table ──────────────────────────────────────────────────
ratio_rows = []
for c in ratio_df.index:
    ratio_rows.append({
        "Compound":                c,
        "DAT/SERT (radioligand)":  fmt_ratio(ratio_df.loc[c, "ratio_radio"],
                                             ratio_df.loc[c, "ratio_radio_CI_lower"],
                                             ratio_df.loc[c, "ratio_radio_CI_upper"]),
        "DAT/SERT (fluorescence)": fmt_ratio(ratio_df.loc[c, "ratio_fluoro"],
                                             ratio_df.loc[c, "ratio_fluoro_CI_lower"],
                                             ratio_df.loc[c, "ratio_fluoro_CI_upper"]),
    })

pd.DataFrame(ratio_rows).to_csv("ratio_table.csv", index=False)
print("Ratio table saved to ratio_table.csv")

y_abs = np.ceil(
    (max(abs(upper3), abs(lower3),
         abs(diff_log.max()), abs(diff_log.min())) + 0.2) * 2) / 2

xba_min = mean_log.min() - 0.3
xba_max = mean_log.max() + 0.3

ax_ba.fill_between([xba_min, xba_max], lower3, upper3,
                   color="lightgrey", alpha=0.5, zorder=1)
ax_ba.scatter(mean_log, diff_log, color="black", zorder=3, s=40)

for name in ratio_df.index:
    xi = mean_log[name]
    yi = diff_log[name]
    dx, dy = ba_offsets.get(name, (5, 4))
    ax_ba.annotate(
        name, (xi, yi),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=7.5, color="black",
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5,
                        shrinkA=2, shrinkB=2)
        if abs(dx) > 10 or abs(dy) > 10 else None)

ax_ba.axhline(md3,    linestyle="--", linewidth=0.8, color="black",
              label=f"Mean = {md3:.2f}")
ax_ba.axhline(upper3, linestyle=":",  linewidth=0.8, color="black",
              label=f"Upper LoA = {upper3:.2f}")
ax_ba.axhline(lower3, linestyle=":",  linewidth=0.8, color="black",
              label=f"Lower LoA = {lower3:.2f}")
ax_ba.axhline(0,      linestyle="-",  linewidth=0.4, color="black")
ax_ba.set_xlim(xba_min, xba_max)
ax_ba.set_ylim(-y_abs, y_abs)

log_ticks = [-1, 0, 1, 2, 3, 4]
ax_ba.set_xticks(log_ticks)
ax_ba.set_xticklabels([f"{10**v:g}" for v in log_ticks])
ax_ba.set_xlabel("Mean selectivity ratio  IC$_{50}$(SERT)/IC$_{50}$(DAT)", fontsize = 12)
ax_ba.set_ylabel("Difference\nlog₁₀(fluorescence) − log₁₀(radioligand)", fontsize = 12)
style_legend(ax_ba.legend(fontsize=11, loc="lower left"))

print(f"\nDAT/SERT ratio Bland–Altman (log scale):")
print(f"  Mean log diff = {md3:.3f}  (≈ {10**md3:.2f}-fold systematic bias)")
print(f"  95% CI        = {ci3[0]:.3f} to {ci3[1]:.3f}")
print(f"  LoA           = {lower3:.3f} to {upper3:.3f}")

fig3.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.92, wspace=0.35)
plt.savefig("ratio_plots.png", dpi=120)