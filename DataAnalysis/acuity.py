#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
acuity.py  ·  Statistical dashboard for ED arrivals & LOS by raw acuity levels

• Console summaries: acuity mix, LOS stats, inter-arrival stats
• Plots: frequency bar, violin/box, density histogram, ECDF,
  overall LOS KDE, inter-arrival KDE, hourly arrival bar,
  one-week Poisson simulation with 95 % CI
• All figures are saved to a ./graphs folder and shown interactively.
• Writes `empirical_params.json` containing:
      - lambda_hour          : list of 24 hourly arrival rates
      - acuity_probs         : list of probabilities for acuity levels
      - lognorm_params       : dict mapping acuity to [mu, sigma]
      - caregroup_los        : nested dict of mean/std LOS by acuity × care group
      - acuity_los_summary   : summary stats per acuity level
      - overall_los_stats    : summary stats across all patients
"""
import pathlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import skew, lognorm

# Styling
rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")

# Paths & constants
ED_PATH          = "unused/edstays.csv/edstays.csv"
TRI_PATH         = "unused/triage.csv/triage.csv"
ADM_PATH         = "unused/admissions.csv/admissions.csv"
TRANSFERS_PATH   = "unused/transfers.csv/transfers.csv"
PATIENTS_PATH    = "unused/patients.csv/patients.csv"
MAX_LOS_H        = 30 * 24    # 30 days in hours
TRIM_PCT         = 0.99       # 99th percentile trim for inter-arrival
TARGET_DAILY_AVG = 70        # desired average arrivals per day


def summary(series: pd.Series) -> dict:
    return {
        "n": int(series.count()),
        "mean": round(series.mean(), 1),
        "median": round(series.median(), 1),
        "sd": round(series.std(ddof=0), 1),
        "p95": round(np.percentile(series, 95), 1),
        "skew": round(skew(series), 2)
    }


def savefig(fig: plt.Figure, stem: str):
    outdir = pathlib.Path(__file__).parent / "acuity"
    outdir.mkdir(parents=True, exist_ok=True)
    filepath = outdir / f"{stem}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"[Saved] acuity/{stem}.png")


def simplify(u: str) -> str:
    m = {
        "Emergency Department": "ED",
        "Emergency Department Observation": "ED Obs",
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro ICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "Cardiac ICU",
        "Intensive Care Unit (ICU)": "ICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "ICU",
        "Med/Surg": "MedSurg",
        "Med/Surg/Trauma": "MedSurg",
        "Med/Surg/GYN": "MedSurg",
        "Medicine/Cardiology": "Cardiology",
        "Coronary Care Unit (CCU)": "Cardiology",
        "Hematology/Oncology": "Oncology",
        "Hematology/Oncology Intermediate": "Onc-Step",
        "Med/Surg/Trauma": "MedSurg",
        "Medicine/Gastroenterology": "GI",
        "General Surgery": "Surg Ward",
        "Surgery": "Surg Ward",
        "Surgery/Trauma": "Surg Ward",
        "Surgery/Vascular/Intermediate": "Surg-Step",
        "Thoracic Surgery": "Surg Ward",
        "Observation": "Obs Unit",
        "Discharge Lounge": "Discharge",
        "UNKNOWN": "Discharge",
        "Unknown": "Discharge",
        "Labor & Delivery": "L&D",
        "Obstetrics Postpartum": "OB Post",
        "Obstetrics Antepartum": "OB Ante",
        "Psychiatry": "Psych",
        "Nursery": "Nursery",
        "Special Care Nursery (SCN)": "Nursery"
    }
    return m.get(u, u[:12])


def group(s: str) -> str:
    if s in ["ED", "ED Obs"]:
        return "Emergency"
    if s in ["MICU", "SICU", "ICU", "Neuro ICU", "Cardiac ICU"]:
        return "ICU"
    if "Step" in s:
        return "Step-Down"
    if s in ["MedSurg", "Cardiology", "Oncology", "GI", "OB Post", "OB Ante"]:
        return "Medical Ward"
    if "Surg" in s:
        return "Surgical Ward"
    if s in ["L&D", "OB Ante", "OB Post"]:
        return "Obstetrics"
    if s in ["Nursery", "SCN"]:
        return "Pediatrics"
    if s == "Psych":
        return "Psychiatry"
    if s == "Discharge":
        return "Discharge"
    if "Obs" in s:
        return "Observation"
    return "Other"


def calculate_lambda_hour(ed_df: pd.DataFrame,
                          patients_df: pd.DataFrame,
                          target_daily_avg: int = TARGET_DAILY_AVG) -> list:
    df = ed_df.dropna(subset=['intime']).copy()
    df['intime'] = pd.to_datetime(df['intime'], errors='coerce')
    if 'disposition' in df.columns:
        df = df[~df['disposition'].isin(['LEFT WITHOUT BEING SEEN','LEFT AMA'])]
    df = df.merge(
        patients_df[['subject_id','anchor_year','anchor_year_group']],
        on='subject_id', how='left'
    )
    df['date'] = df['intime'].dt.date
    df['hour'] = df['intime'].dt.hour
    active_days = (df['date'].value_counts() > 0).sum()
    counts = df['hour'].value_counts().sort_index().reindex(range(24), fill_value=0)
    raw_lambda = counts / active_days
    scaling = target_daily_avg / raw_lambda.sum()
    return (raw_lambda * scaling).round(2).tolist()


def run():
    # 1) Load data
    tri = pd.read_csv(TRI_PATH)
    ed  = pd.read_csv(ED_PATH, parse_dates=["intime","outtime"], dayfirst=True)
    adm = pd.read_csv(ADM_PATH, parse_dates=["admittime","dischtime"])
    try:
        trans = pd.read_csv(TRANSFERS_PATH)
    except FileNotFoundError:
        trans = None
        print("[Warn] transfers.csv not found; skipping careunit grouping.")
    patients = pd.read_csv(PATIENTS_PATH)

    # 2) Merge triage + ED + admission data
    tri['acuity'] = pd.to_numeric(tri['acuity'], errors='coerce')
    tri = tri.drop_duplicates(['subject_id','stay_id'], keep='first')
    edk = ed[['subject_id','stay_id','hadm_id','intime']].dropna(subset=['hadm_id'])
    df = tri.merge(edk, on=['subject_id','stay_id'])
    df = df.merge(
        adm[['subject_id','hadm_id','admittime','dischtime']],
        on=['subject_id','hadm_id'], how='left'
    )
    df = df.dropna(subset=['admittime','dischtime','acuity'])
    df['hospital_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    df = df[df['hospital_hours'] > 0]
    df['acuity_level'] = df['acuity'].astype(int)

    # 3) Acuity mix & LOS distribution fits
    acuity_probs = df['acuity_level'].value_counts(normalize=True).sort_index()
    trimmed = df[df['hospital_hours'] < MAX_LOS_H]
    fitted = {}
    for lvl, sub in trimmed.groupby('acuity_level'):
        shape, loc, scale = lognorm.fit(sub['hospital_hours'], floc=0)
        fitted[int(lvl)] = [float(np.log(scale)), float(shape)]

    # 3b) LOS summaries
    acuity_los_summary = {int(lvl): summary(sub['hospital_hours'])
                           for lvl, sub in df.groupby('acuity_level')}
    overall_los_stats = summary(df['hospital_hours'])

    # 4) Inter-arrival times
    eds = ed.dropna(subset=['intime']).copy()
    eds['intime'] = pd.to_datetime(eds['intime'], errors='coerce')
    ia = eds['intime'].diff().dt.total_seconds().div(60).dropna()

    # 5) Hourly arrival rates
    lambda_hour = calculate_lambda_hour(ed, patients)

    # 6) LOS by acuity × care group
    if trans is not None and 'careunit' not in df.columns:
        if 'transfertime' in trans.columns:
            trans['transfertime'] = pd.to_datetime(trans['transfertime'], errors='coerce')
        first_trans = trans.sort_values(
            'transfertime' if 'transfertime' in trans.columns else trans.columns[0]
        ).groupby('hadm_id')['careunit'].first()
        df['careunit'] = df['hadm_id'].map(first_trans)
    elif 'careunit' in ed.columns:
        df = df.merge(ed[['subject_id','stay_id','careunit']], on=['subject_id','stay_id'], how='left')

    df['unit_simplified'] = df['careunit'].map(simplify)
    df['care_group'] = df['unit_simplified'].map(group)
    df['hospital_minutes'] = df['hospital_hours'] * 60
    grp_mean = df.groupby(['acuity_level','care_group'])['hospital_minutes'].mean().unstack(fill_value=0).round(1)
    grp_std = df.groupby(['acuity_level','care_group'])['hospital_minutes'].std(ddof=0).unstack(fill_value=0).round(1)

    # Save CSVs
    outdir = pathlib.Path(__file__).parent / 'acuity'
    outdir.mkdir(parents=True, exist_ok=True)
    grp_mean.to_csv(outdir / 'los_by_acuity_and_caregroup_mean.csv')
    grp_std.to_csv(outdir  / 'los_by_acuity_and_caregroup_std.csv')
    print(f"[Saved] los_by_acuity_and_caregroup_mean.csv, los_by_acuity_and_caregroup_std.csv")

    # 7) Figures 01-06: acuity, LOS, ECDF, KDE
    fig, ax = plt.subplots(figsize=(6,4))
    acuity_probs.plot(kind='bar', ax=ax)
    ax.set(title='Frequency by Acuity Level', xlabel='Acuity Level', ylabel='Proportion')
    savefig(fig,'fig01_acuity_frequency')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.violinplot(data=trimmed, x='acuity_level', y='hospital_hours', inner='box', ax=ax)
    ax.set(ylim=(0,MAX_LOS_H), title='LOS by Acuity Level (<30 days)', ylabel='Hospital LOS (h)')
    savefig(fig,'fig02_los_violin')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(trimmed, x='hospital_hours', hue='acuity_level', element='step',
                 stat='density', common_norm=False, binwidth=6, ax=ax)
    ax.set(xlim=(0,MAX_LOS_H), title='Density of LOS by Acuity Level (<30 days)')
    savefig(fig,'fig03_los_density')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    for lvl, sub in trimmed.groupby('acuity_level'):
        x = np.sort(sub['hospital_hours'])
        y = np.linspace(0,1,len(x))
        ax.step(x,y,label=str(lvl))
    ax.set(xlim=(0,MAX_LOS_H), title='ECDF of LOS by Acuity Level (<30 days)')
    ax.legend(title='Acuity')
    savefig(fig,'fig04_los_ecdf')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(trimmed['hospital_hours'], fill=True, ax=ax)
    ax.set(xlim=(0,MAX_LOS_H), title='Overall LOS distribution (<30 days)')
    savefig(fig,'fig05_los_kde')
    plt.show()

    ia_trim = ia[ia < ia.quantile(TRIM_PCT)]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.kdeplot(ia_trim, fill=True, ax=ax)
    ax.set(title='ED inter-arrival KDE (<99th pct)')
    savefig(fig,'fig06_interarrival_kde')
    plt.show()

    # 8) Hourly arrival bar
    fig, ax = plt.subplots(figsize=(8,5))
    pd.Series(lambda_hour, index=range(24)).plot(kind='bar', ax=ax)
    ax.set(title='Hourly arrival rate', xlabel='Hour of Day', ylabel='Patients/hr')
    savefig(fig,'fig07_hourly_arrivals')
    plt.show()

    # 9) One-week Poisson simulation
    wc = np.random.poisson(lam=lambda_hour, size=(7,24))
    fig, ax = plt.subplots(figsize=(10,5))
    for d in range(7): ax.plot(range(24), wc[d], marker='o', label=f'Day {d+1}')
    ax.set(title='One-week Poisson simulation')
    ax.legend(ncol=7, fontsize=7)
    savefig(fig,'fig08_poisson_sim')
    plt.show()

    # 10) Poisson 95% CI
    mean_week = wc.mean(axis=0)
    se_week   = wc.std(axis=0, ddof=1) / np.sqrt(7)
    ci95      = 1.96 * se_week
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(24), mean_week)
    ax.errorbar(range(24), mean_week, yerr=ci95, fmt='none', capsize=3)
    ax.set(title='Expected hourly arrivals with 95% CI')
    savefig(fig,'fig09_poisson_ci')
    plt.show()

    # 11) Empirical parameters JSON
    artefacts = {
        'lambda_hour':         lambda_hour,
        'acuity_probs':        acuity_probs.sort_index().tolist(),
        'lognorm_params':      fitted,
        'caregroup_los':       {str(acu): {care: {'mean': float(grp_mean.loc[acu,care]), 'std': float(grp_std.loc[acu,care])}
                                  for care in grp_mean.columns}
                                 for acu in grp_mean.index},
        'acuity_los_summary':  acuity_los_summary,
        'overall_los_stats':   overall_los_stats
    }
    json_path = pathlib.Path(__file__).parent / 'empirical_params.json'
    with open(json_path, 'w') as f:
        json.dump(artefacts, f, indent=2)
    print(f"[Saved] {json_path.name}")

if __name__ == '__main__':
    run()
