#!/usr/bin/env python
# transfers.py  –  FULLY REFRACTORED (with Data Cleaning, Stats & Full Plots)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # only for the weekly trend

# -------------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------------
CSV_PATH = "unused/transfers.csv/transfers.csv"   # raw transfer file
OUT_DIR  = "DataAnalysis/graph"                  # output directory
THR_MOVE = 0.03                                   # hide moves < 3% in heat‑map
THR_OTHER= 0.02                                   # if “Other” <2% keep, else split
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. LOAD & BASIC CLEAN
# -------------------------------------------------------------------
df = (
    pd.read_csv(CSV_PATH, parse_dates=["intime","outtime"])
      .dropna(subset=["intime","eventtype"])
)
# compute stay duration in minutes
df["duration_minutes"] = (
    (df["outtime"].dt.round("s") - df["intime"].dt.round("s"))
    .dt.total_seconds() / 60
)
# keep only 1 minute to 30 days
df = df[df["duration_minutes"].between(1, 43200)]
# extract time components
df["hour"] = df["intime"].dt.hour
df["date"] = df["intime"].dt.date
df["week"] = df["intime"].dt.to_period("W").dt.start_time

# -------------------------------------------------------------------
# 2. MAP ORIGINAL UNIT → SIMPLE → GROUP
# -------------------------------------------------------------------
def simplify(u):
    m = {
        "Emergency Department":"ED", "Emergency Department Observation":"ED Obs",
        "Medical Intensive Care Unit (MICU)":"MICU", "Surgical Intensive Care Unit (SICU)":"SICU",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)":"Neuro ICU", "Cardiac Vascular Intensive Care Unit (CVICU)":"Cardiac ICU",
        "Intensive Care Unit (ICU)":"ICU", "Medical/Surgical Intensive Care Unit (MICU/SICU)":"ICU",
        "Med/Surg":"MedSurg", "Med/Surg/Trauma":"MedSurg", "Med/Surg/GYN":"MedSurg",
        "Medicine/Cardiology":"Cardiology", "Coronary Care Unit (CCU)":"Cardiology",
        "Hematology/Oncology":"Oncology", "Hematology/Oncology Intermediate":"Onc‑Step",
        "Neuro Intermediate":"Neuro‑Step", "Medical/Surgical (Gynecology)":"GYN‑Step",
        "Transplant":"Tx Ward", "Surgery":"Surg Ward", "Surgery/Trauma":"Surg Ward",
        "Surgery/Vascular/Intermediate":"Surg‑Step", "Thoracic Surgery":"Surg Ward",
        "Observation":"Obs Unit", "Discharge Lounge":"Discharge",
        "UNKNOWN":"Discharge", "Unknown":"Discharge",
        "Labor & Delivery":"L&D", "Obstetrics Postpartum":"OB Post",
        "Obstetrics Antepartum":"OB Ante", "Psychiatry":"Psych",
        "Nursery":"Nursery", "Special Care Nursery (SCN)":"Nursery"
    }
    return m.get(u, u[:12])

def group(s):
    if s in ["ED","ED Obs"]:
        return "Emergency"
    if s in ["MICU","SICU","ICU","Neuro ICU","Cardiac ICU"]:
        return "ICU"
    if "Step" in s:
        return "Step-Down"
    if s in ["MedSurg","Cardiology","Oncology","Tx Ward","OB Post","OB Ante"]:
        return "Medical Ward"
    if "Surg" in s:
        return "Surgical Ward"
    if s in ["L&D","OB Ante","OB Post"]:
        return "Obstetrics"
    if s in ["Nursery","SCN"]:
        return "Pediatrics"
    if s == "Psych":
        return "Psychiatry"
    if s == "Discharge":
        return "Discharge"
    if "Obs" in s:
        return "Observation"
    return "Other"

# apply mappings
df["simple"] = df["careunit"].apply(simplify)
df["group"]  = df["simple"].apply(group)
# expand "Other" if large
top_share = (df["group"]=="Other").mean()
if top_share > THR_OTHER:
    top3 = df.loc[df["group"]=="Other","simple"].value_counts().head(3).index.tolist()
    df.loc[df["simple"].isin(top3), "group"] = df["simple"]
    df.loc[df["group"]=="Other", "group"] = "Other-Small"
# save mapping
(df[["careunit","simple","group"]]
   .drop_duplicates()
   .to_csv(os.path.join(OUT_DIR,"unit_mapping.csv"), index=False)
)

# -------------------------------------------------------------------
# 3. DATA CLEANING (robust outlier removal per group)
#    Uses IQR-based trimming with optional percentile fallback
# -------------------------------------------------------------------

def clean_by_iqr(frame, col="duration_minutes", k=1.5):
    """
    Trim values outside [Q1 - k*IQR, Q3 + k*IQR] per group.
    """
    cleaned = []
    summary = []
    for g, sub in frame.groupby("group"):
        q1 = sub[col].quantile(0.25)
        q3 = sub[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        kept = sub[sub[col].between(lo, hi)]
        # if too few remain, fallback to percentile trimming
        if len(kept) < max(5, 0.5 * len(sub)):
            # fallback to 1st-99th percentile
            p_lo, p_hi = sub[col].quantile([0.01, 0.99])
            kept = sub[sub[col].between(p_lo, p_hi)]
            method = "percentile"
        else:
            method = "iqr"
        cleaned.append(kept)
        summary.append((g, len(sub), len(kept), method))
    # report per-group retention
    print("🔍 Cleaning summary per group:")
    for g, total, kept, method in summary:
        print(f"  {g}: kept {kept}/{total} ({kept/total:.1%}), method={method}")
    return pd.concat(cleaned)

# apply robust cleaning
df_clean = clean_by_iqr(df)
print(f"🔍 After cleaning: kept {len(df_clean)}/{len(df)} rows ({len(df_clean)/len(df):.1%})")
# -------------------------------------------------------------------
# 4. STATISTICS: RAW vs CLEANED
# -------------------------------------------------------------------
# 4. STATISTICS: RAW vs CLEANED
# -------------------------------------------------------------------
def make_stats(frame):
    return (
        frame.groupby("group")["duration_minutes"]
             .agg(
                 avg_duration_min="mean",
                 var_duration_min="var",
                 median_min="median",
                 mad_min=lambda x: np.median(np.abs(x - x.median()))
             )
             .reset_index()
    )

stats_raw   = make_stats(df)
stats_clean = make_stats(df_clean)

stats_raw.to_csv(os.path.join(OUT_DIR,"duration_stats_raw.csv"),    index=False)
stats_clean.to_csv(os.path.join(OUT_DIR,"duration_stats_clean.csv"), index=False)

print("✅ Raw duration stats")
print(stats_raw)
print("✅ Cleaned duration stats")
print(stats_clean)

# -------------------------------------------------------------------
# 5. PLOTS: RAW vs CLEANED GROUP DURATION
# -------------------------------------------------------------------
plt.figure(figsize=(10,5))
sns.barplot(
    data=stats_raw.sort_values("avg_duration_min", ascending=False),
    x="group", y="avg_duration_min"
)
plt.xticks(rotation=45, ha="right")
plt.title("Raw Avg Duration per Care Unit Group")
plt.xlabel("Group")
plt.ylabel("Avg Duration (min)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"avg_duration_group_raw.png"))
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(
    data=stats_clean.sort_values("avg_duration_min", ascending=False),
    x="group", y="avg_duration_min"
)
plt.xticks(rotation=45, ha="right")
plt.title("Cleaned Avg Duration per Care Unit Group")
plt.xlabel("Group")
plt.ylabel("Avg Duration (min)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"avg_duration_group_clean.png"))
plt.show()

# -------------------------------------------------------------------
# 6. WITHIN‑STAY & EXIT TABLES
# -------------------------------------------------------------------
chron = df.sort_values(["subject_id","hadm_id","intime"])
chron["next_group"] = chron.groupby(["subject_id","hadm_id"])["group"].shift(-1)

moves = chron[chron["group"]!="Discharge"].dropna(subset=["next_group"])
M = moves.groupby(["group","next_group"]).size().unstack(fill_value=0)
P_move = M.div(M.sum(axis=1), axis=0)
P_move.to_csv(os.path.join(OUT_DIR,"P_within_stay.csv"))

terminal = chron.groupby(["subject_id","hadm_id"]).tail(1)
P_exit = terminal["group"].value_counts(normalize=True)
P_exit.to_csv(os.path.join(OUT_DIR,"P_exit_by_group.csv"))
print("✅ Transition & exit CSVs written to", OUT_DIR)

# -------------------------------------------------------------------
# 7. OTHER PLOTS (unchanged)
# -------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# ED arrivals per hour
plt.figure(figsize=(10,4))
sns.histplot(df[df["eventtype"]=="ED"]["hour"], bins=24, edgecolor="black")
plt.title("ED Arrivals per Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"ed_arrivals_hourly.png"))
plt.show()

# Avg ED arrivals per hour (per day)
ed_unique = df[df["eventtype"]=="ED"].drop_duplicates(subset=["subject_id","intime"])
mu = (
    ed_unique.groupby(["date","hour"]).size()
             .reset_index(name="cnt")
             .groupby("hour")["cnt"].mean()
)
plt.figure(figsize=(10,5))
sns.barplot(x=mu.index, y=mu.values)
plt.title("Average ED Arrivals per Hour (per day)")
plt.xlabel("Hour")
plt.ylabel("Avg count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"ed_arrivals_normalised.png"))
plt.show()

# Weekly transfers trend
wk = df.groupby(["week","group"]).size().reset_index(name="count")
fig = px.line(wk, x="week", y="count", color="group", title="Weekly Transfers by Domain")
fig.write_html(os.path.join(OUT_DIR,"weekly_transfers.html"))
fig.show()

# Event duration boxplots
clip = df["duration_minutes"].quantile(0.99)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))
sns.boxplot(data=df[df["duration_minutes"]<clip], x="eventtype", y="duration_minutes", ax=ax1)
ax1.set_title("Durations per Event (linear, <99th)")
ax1.set_ylabel("minutes")
sns.boxplot(data=df, x="eventtype", y="duration_minutes", ax=ax2)
ax2.set_title("Durations per Event (log)")
ax2.set_yscale("log")
ax2.set_ylabel("minutes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"event_durations.png"))
plt.show()

# Within-stay heatmap
import numpy as _np  # ensure alias for mask code
diag = _np.zeros_like(P_move, dtype=bool)
for i, row_label in enumerate(P_move.index):
    if row_label in P_move.columns:
        diag[i, P_move.columns.get_loc(row_label)] = True
mask = pd.DataFrame(diag, index=P_move.index, columns=P_move.columns)
H = P_move.mask(mask).where(P_move >= THR_MOVE)
plt.figure(figsize=(9,6))
sns.heatmap(H, annot=True, fmt=".02f", cmap="Blues", linewidths=.4, cbar_kws=dict(label="P(next | current)"))
plt.title(f"Within‑Stay Transitions (≥{int(THR_MOVE*100)}%, self‑loops hidden)")
plt.xlabel("Next Group")
plt.ylabel("Current Group")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"within_stay_heatmap.png"))
plt.show()

# Exit probabilities
plt.figure(figsize=(7,4))
P_exit.sort_values(ascending=False).plot(kind="bar", edgecolor="black")
plt.title("Exit Probability by Current Group")
plt.ylabel("P(exit)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"exit_probs.png"))
plt.show()

# Other-Small breakdown
if "Other-Small" in df["group"].unique():
    top_other = df.loc[df["group"]=="Other-Small","simple"].value_counts().head(10)
    plt.figure(figsize=(8,4))
    top_other.plot(kind="bar", edgecolor="black")
    plt.title("Top constituents inside 'Other-Small'")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"other_small_breakdown.png"))
    plt.show()
