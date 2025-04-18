#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
acuity.py  ·  Statistical dashboard + parameter export for ED arrivals & LOS

• Console summaries: acuity mix, LOS stats, inter‑arrival stats
• Plots: frequency bar, violin/box, density histogram, ECDF, overall LOS KDE,
  inter‑arrival KDE, hourly arrival bar, one‑week Poisson simulation with 95 % CI.
• All figures are saved to a ./graphs folder (created if missing) and shown interactively.
• Writes `empirical_params.json` containing:
      - lambda_hour    : list of 24 hourly arrival rates
      - class_probs    : list of policy-group probabilities [G1-2, G3, G4-5]
      - lognorm_params : dict mapping group to [mu, sigma]
"""

import pathlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import skew, lognorm, kruskal

# Use a font that supports non-breaking hyphens
rcParams["font.family"] = "DejaVu Sans"

# ----------------------------------------------------------------------------
# User paths – adjust if CSVs are located elsewhere
ED_PATH  = "unused/edstays.csv/edstays.csv"
TRI_PATH = "unused/triage.csv/triage.csv"
ADM_PATH = "unused/admissions.csv/admissions.csv"
# ----------------------------------------------------------------------------
MAX_LOS_H    = 30 * 24      # trim LOS plots to 30 days
TRIM_PCT     = 0.99         # trim inter‑arrival KDE
POLICY_GROUP = {1: "G1-2", 2: "G1-2", 3: "G3", 4: "G4-5", 5: "G4-5"}

sns.set_style("whitegrid")


def summary(series: pd.Series) -> dict:
    """Return quick descriptive stats for a numeric Series."""
    return {
        "n":      int(series.count()),
        "mean":   round(series.mean(), 1),
        "median": round(series.median(), 1),
        "sd":     round(series.std(ddof=0), 1),
        "p95":    round(np.percentile(series, 95), 1),
        "skew":   round(skew(series), 2)
    }


def savefig(fig: plt.Figure, stem: str):
    """
    Save a Matplotlib figure into a graphs/ folder next to this script.
    """
    base_dir = pathlib.Path(__file__).parent
    gdir = base_dir / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] graphs/{path.name}")


def run(ed_path=ED_PATH, tri_path=TRI_PATH, adm_path=ADM_PATH):
    # 1) Load data
    tri = pd.read_csv(tri_path)
    ed  = pd.read_csv(ed_path,  parse_dates=["intime","outtime"], dayfirst=True)
    adm = pd.read_csv(adm_path, parse_dates=["admittime","dischtime"])

    # 2) Clean triage
    tri["acuity"] = pd.to_numeric(tri["acuity"], errors="coerce")
    tri = tri.sort_values("acuity").drop_duplicates(["subject_id","stay_id"])

    # 3) Merge keys
    key_cols = ["subject_id","stay_id","hadm_id","intime"]
    ed_key   = ed[key_cols].dropna(subset=["hadm_id"])
    df = (tri.merge(ed_key, on=["subject_id","stay_id"]) 
              .merge(adm[["subject_id","hadm_id","admittime","dischtime"]],
                     on=["subject_id","hadm_id"]))

    # 4) Compute LOS
    df = df.dropna(subset=["admittime","dischtime","acuity"])
    df["hospital_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds()/3600
    df = df[df["hospital_hours"]>0]
    df["acuity_group"] = df["acuity"].map(POLICY_GROUP)

    # 5) Console summaries
    print("\n=== Policy-group probabilities ===")
    class_probs = df["acuity_group"].value_counts(normalize=True).sort_index()
    print(class_probs.to_string())

    print("\n=== LOS stats by raw acuity (1-5) ===")
    los_raw = pd.DataFrame({a: summary(g["hospital_hours"]) 
                             for a,g in df.groupby("acuity")}).T
    print(los_raw.to_string())

    trimmed = df[df["hospital_hours"]<MAX_LOS_H]
    print("\n=== LOS stats by policy group (<30 days) ===")
    los_grp = pd.DataFrame({g: summary(s["hospital_hours"]) 
                             for g,s in trimmed.groupby("acuity_group")}).T
    print(los_grp.to_string())

    print("\n--- Log-normal μ,σ fits (<30 days) ---")
    fitted_dict = {}
    for g, sub in trimmed.groupby("acuity_group"):
        shape, loc, scale = lognorm.fit(sub["hospital_hours"], floc=0)
        mu, sigma = np.log(scale), shape
        fitted_dict[g] = (mu, sigma)
        print(f"{g:>5}: mu={mu:.3f}, sigma={sigma:.3f}, n={len(sub)}")

    kw = kruskal(*[s["hospital_hours"] for _,s in trimmed.groupby("acuity_group")])
    print(f"\n[Kruskal-Wallis] H={kw.statistic:.1f}, p={kw.pvalue:.3g}")

    # 6) Inter-arrival stats
    ed_sorted = ed.sort_values("intime").assign(intime=pd.to_datetime(ed["intime"]))
    ia = ed_sorted["intime"].diff().dt.total_seconds().div(60).dropna()
    print("\n=== Inter-arrival descriptive (minutes) ===")
    print(ia.describe()[["count","mean","std","50%","min","max"]].round(1).to_string())

    ed_sorted["hour"] = ed_sorted["intime"].dt.hour
    ed_sorted["date"] = ed_sorted["intime"].dt.date
    n_days = ed_sorted["date"].nunique()
    lambda_hour = ed_sorted["hour"].value_counts().sort_index() / n_days

    # 7) Plot 7 figures
    # 7.1 Frequency bar
    fig1, ax1 = plt.subplots(figsize=(6,4))
    class_probs.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Proportion of arrivals")
    ax1.set_title("Policy-group frequency")
    fig1.tight_layout()
    savefig(fig1, "fig01_policy_group_frequency")
    plt.show()

    # 7.2 Violin + box LOS
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.violinplot(data=trimmed, x="acuity_group", y="hospital_hours",
                   inner="box", ax=ax2)
    ax2.set_ylim(0, MAX_LOS_H)
    ax2.set_ylabel("Hospital LOS (h)")
    ax2.set_title("LOS by acuity (<30 days)")
    fig2.tight_layout()
    savefig(fig2, "fig02_los_violin")
    plt.show()

    # 7.3 Density histogram per group
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.histplot(data=trimmed, x="hospital_hours", hue="acuity_group",
                 element="step", stat="density", common_norm=False,
                 binwidth=6, ax=ax3)
    ax3.set_xlim(0, MAX_LOS_H)
    ax3.set_xlabel("Hospital LOS (h)")
    ax3.set_title("Density of LOS by acuity (<30 days)")
    fig3.tight_layout()
    savefig(fig3, "fig03_los_density")
    plt.show()

    # 7.4 ECDF
    fig4, ax4 = plt.subplots(figsize=(8,5))
    for g, sub in trimmed.groupby("acuity_group"):
        x = np.sort(sub["hospital_hours"])
        y = np.linspace(0,1,len(x))
        ax4.step(x, y, label=g)
    ax4.set_xlim(0, MAX_LOS_H)
    ax4.set_xlabel("Hospital LOS (h)")
    ax4.set_ylabel("ECDF")
    ax4.set_title("ECDF of LOS by acuity (<30 days)")
    ax4.legend()
    fig4.tight_layout()
    savefig(fig4, "fig04_los_ecdf")
    plt.show()

    # 7.5 Overall LOS KDE
    fig5, ax5 = plt.subplots(figsize=(8,5))
    sns.kdeplot(trimmed["hospital_hours"], fill=True, ax=ax5)
    ax5.set_xlim(0, MAX_LOS_H)
    ax5.set_xlabel("Hospital LOS (h)")
    ax5.set_title("Overall LOS distribution (<30 days)")
    fig5.tight_layout()
    savefig(fig5, "fig05_los_kde")
    plt.show()

    # 7.6 Inter-arrival KDE
    ia_trim = ia[ia < ia.quantile(TRIM_PCT)]
    fig6, ax6 = plt.subplots(figsize=(8,4))
    sns.kdeplot(ia_trim, fill=True, ax=ax6)
    ax6.set_xlabel("Inter-arrival time (min)")
    ax6.set_title("ED inter-arrival KDE (<99th pct)")
    fig6.tight_layout()
    savefig(fig6, "fig06_interarrival_kde")
    plt.show()

    # 7.7 Hourly arrival bar
    fig7, ax7 = plt.subplots(figsize=(8,4))
    ed_sorted["hour"].value_counts().sort_index().plot(
        kind="bar", ax=ax7)
    ax7.set_xlabel("Hour of day")
    ax7.set_ylabel("Arrivals")
    ax7.set_title("Hourly arrival frequency")
    fig7.tight_layout()
    savefig(fig7, "fig07_hourly_arrivals")
    plt.show()

    # 7.8 One-week Poisson simulation
    lam_vec = lambda_hour.reindex(range(24)).fillna(0).values
    sim_days = 7
    weekly_counts = np.random.poisson(lam=lam_vec, size=(sim_days,24))
    fig8, ax8 = plt.subplots(figsize=(10,5))
    for d in range(sim_days):
        ax8.plot(range(24), weekly_counts[d], marker="o", label=f"Day {d+1}")
    ax8.set_xticks(range(24))
    ax8.set_xlabel("Hour of day")
    ax8.set_ylabel("Arrivals")
    ax8.set_title("One-week Poisson simulation of hourly arrivals")
    ax8.legend(ncol=7, fontsize=7)
    fig8.tight_layout()
    savefig(fig8, "fig08_poisson_sim")
    plt.show()

    # 7.9 Mean ±95% CI bar
    mean_week = weekly_counts.mean(axis=0)
    se_week   = weekly_counts.std(axis=0, ddof=1) / np.sqrt(sim_days)
    ci95      = 1.96 * se_week
    fig9, ax9 = plt.subplots(figsize=(8,4))
    ax9.bar(range(24), mean_week)
    ax9.errorbar(range(24), mean_week, yerr=ci95, fmt="none", capsize=3)
    ax9.set_xlabel("Hour of day")
    ax9.set_ylabel("Mean arrivals (7-day sim)")
    ax9.set_title("Expected hourly arrivals with 95% CI")
    fig9.tight_layout()
    savefig(fig9, "fig09_poisson_ci")
    plt.show()

    return lambda_hour, class_probs, fitted_dict


# ------------------------------ CLI entry ------------------------------ #
if __name__ == "__main__":
    lam24, class_probs, fitted = run()
    artefacts = {
        "lambda_hour":    lam24.reindex(range(24)).fillna(0).tolist(),
        "class_probs":    class_probs.sort_index().tolist(),
        "lognorm_params": {g: [float(mu), float(sigma)] for g, (mu, sigma) in fitted.items()}
    }
    outpath = pathlib.Path(__file__).parent / "empirical_params.json"
    outpath.write_text(json.dumps(artefacts, indent=2))
    print(f"[Saved] {outpath.name}")
