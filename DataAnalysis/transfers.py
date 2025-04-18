#!/usr/bin/env python
# transfers.py  –  FINAL (with Discharge excluded from move‑matrix)

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px                              # only for the weekly trend

# -------------------------------------------------------------------
# 0. CONFIG
# -------------------------------------------------------------------
CSV_PATH   = "unused/transfers.csv/transfers.csv"   # raw transfer file
OUT_DIR    = "DataAnalysis/graph"                   # where all artefacts go
THR_MOVE   = 0.03                                   # hide moves < 3 % in heat‑map
THR_OTHER  = 0.02                                   # if “Other” <2 % keep, else split
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. LOAD & BASIC CLEAN
# -------------------------------------------------------------------
df = (pd.read_csv(CSV_PATH, parse_dates=["intime","outtime"])
        .dropna(subset=["intime","eventtype"]))

df["duration_minutes"] = (df["outtime"].dt.round("s") -
                          df["intime"].dt.round("s")
                         ).dt.total_seconds()/60
df = df[df["duration_minutes"].between(1, 43200)]        # 1 min‑30 days
df["hour"] = df["intime"].dt.hour
df["date"] = df["intime"].dt.date
df["week"] = df["intime"].dt.to_period("W").dt.start_time

# -------------------------------------------------------------------
# 2. MAP ORIGINAL UNIT  →  SIMPLE  →  GROUP
# -------------------------------------------------------------------
def simplify(u):
    m = {
        # -- ED family
        "Emergency Department":"ED","Emergency Department Observation":"ED Obs",
        # -- ICU tiers
        "Medical Intensive Care Unit (MICU)":"MICU",
        "Surgical Intensive Care Unit (SICU)":"SICU",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)":"Neuro ICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)":"Cardiac ICU",
        "Intensive Care Unit (ICU)":"ICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)":"ICU",
        # -- wards & step‑downs
        "Med/Surg":"MedSurg","Med/Surg/Trauma":"MedSurg","Med/Surg/GYN":"MedSurg",
        "Medicine/Cardiology":"Cardiology","Coronary Care Unit (CCU)":"Cardiology",
        "Hematology/Oncology":"Oncology","Hematology/Oncology Intermediate":"Onc‑Step",
        "Neuro Intermediate":"Neuro‑Step","Medical/Surgical (Gynecology)":"GYN‑Step",
        "Transplant":"Tx Ward",
        # -- surgery wards
        "Surgery":"Surg Ward","Surgery/Trauma":"Surg Ward",
        "Surgery/Vascular/Intermediate":"Surg‑Step","Thoracic Surgery":"Surg Ward",
        # -- misc
        "Observation":"Obs Unit","Discharge Lounge":"Discharge",
        "UNKNOWN":"Discharge","Unknown":"Discharge",
        "Labor & Delivery":"L&D","Obstetrics Postpartum":"OB Post",
        "Obstetrics Antepartum":"OB Ante",
        "Psychiatry":"Psych","Nursery":"Nursery","Special Care Nursery (SCN)":"Nursery"
    }
    return m.get(u, u[:12])

df["simple"] = df["careunit"].apply(simplify)

def group(s):
    if s in ["ED","ED Obs"]:                                  return "Emergency"
    if s in ["MICU","SICU","ICU","Neuro ICU","Cardiac ICU"]:  return "ICU"
    if "Step" in s:                                           return "Step‑Down"
    if s in ["MedSurg","Cardiology","Oncology","Tx Ward","OB Post","OB Ante"]:
        return "Medical Ward"
    if "Surg" in s:                                           return "Surgical Ward"
    if "Obs" in s:                                            return "Observation"
    if s == "Discharge":                                      return "Discharge"
    return "Other"

df["group"] = df["simple"].apply(group)

# OPTIONAL – expand “Other” if it hides something large
other_share = (df["group"] == "Other").mean()
if other_share > THR_OTHER:
    # break “Other” into its top 3 constituents + keep tail as Other‑Small
    top3 = (df.loc[df["group"]=="Other","simple"]
              .value_counts().head(3).index.tolist())
    df.loc[df["simple"].isin(top3), "group"] = df["simple"]
    df.loc[df["group"]=="Other",  "group"]   = "Other‑Small"

# save mapping for transparency
(df[["careunit","simple","group"]]
   .drop_duplicates()
   .to_csv(os.path.join(OUT_DIR,"unit_mapping.csv"), index=False))

# -------------------------------------------------------------------
# 3. WITHIN‑STAY & EXIT TABLES
# -------------------------------------------------------------------
chron = df.sort_values(["subject_id","hadm_id","intime"])
chron["next_group"] = chron.groupby(["subject_id","hadm_id"])["group"].shift(-1)

moves = chron[chron["group"]!="Discharge"].dropna(subset=["next_group"])
M = moves.groupby(["group","next_group"]).size().unstack(fill_value=0)
P_move = M.div(M.sum(axis=1), axis=0)           # rows sum to 1
P_move.to_csv(os.path.join(OUT_DIR,"P_within_stay.csv"))

terminal = chron.groupby(["subject_id","hadm_id"]).tail(1)
exit_counts = terminal["group"].value_counts()
P_exit = exit_counts/exit_counts.sum()
P_exit.to_csv(os.path.join(OUT_DIR,"P_exit_by_group.csv"))

print("✅  CSVs written to", OUT_DIR)

# -------------------------------------------------------------------
# 4. PLOTS
# -------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# ED arrivals (raw + mean‑per‑day) ----------------------------------
plt.figure(figsize=(10,4))
sns.histplot(df[df["eventtype"]=="ED"]["hour"], bins=24,
             color="orange", edgecolor="black")
plt.title("ED Arrivals per Hour"); plt.xlabel("Hour"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"ed_arrivals_hourly.png")); plt.show()

ed_unique = df[df["eventtype"]=="ED"].drop_duplicates(subset=["subject_id","intime"])
ed_unique["hour"] = ed_unique["intime"].dt.hour
mu = (ed_unique.groupby(["date","hour"]).size()
               .reset_index(name="cnt")
               .groupby("hour")["cnt"].mean())
plt.figure(figsize=(10,5))
sns.barplot(x=mu.index, y=mu.values, color="steelblue")
plt.title("Average ED Arrivals per Hour (per day)")
plt.xlabel("Hour"); plt.ylabel("Avg count"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"ed_arrivals_normalised.png")); plt.show()

# Weekly transfers ----------------------------------------------------
wk = df.groupby(["week","group"]).size().reset_index(name="count")
fig = px.line(wk, x="week", y="count", color="group",
              title="Weekly Transfers by Domain")
fig.write_html(os.path.join(OUT_DIR,"weekly_transfers.html"))
fig.show()

# Event durations -----------------------------------------------------
clip = df["duration_minutes"].quantile(0.99)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))
sns.boxplot(data=df[df["duration_minutes"]<clip], x="eventtype", y="duration_minutes", ax=ax1)
ax1.set_title("Durations per Event (linear, <99 th)"); ax1.set_ylabel("minutes")
sns.boxplot(data=df, x="eventtype", y="duration_minutes", ax=ax2)
ax2.set_title("Durations per Event (log)"); ax2.set_yscale("log"); ax2.set_ylabel("minutes")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"event_durations.png")); plt.show()

# Within‑stay heat‑map -----------------------------------------------
diag = np.zeros_like(P_move, dtype=bool)
for i, row_label in enumerate(P_move.index):
    if row_label in P_move.columns:
        j = P_move.columns.get_loc(row_label)
        diag[i, j] = True
mask = pd.DataFrame(diag, index=P_move.index, columns=P_move.columns)
H = P_move.mask(mask).where(P_move >= THR_MOVE)
plt.figure(figsize=(9,6))
sns.heatmap(H, annot=True, fmt=".02f", cmap="Blues", linewidths=.4,
            cbar_kws=dict(label="P(next | current)"))
plt.title(f"Within‑Stay Transitions (≥{int(THR_MOVE*100)} %, self‑loops hidden)")
plt.xlabel("Next Group"); plt.ylabel("Current Group")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"within_stay_heatmap.png")); plt.show()

# Exit probabilities --------------------------------------------------
plt.figure(figsize=(7,4))
P_exit.sort_values(ascending=False).plot(kind="bar", color="seagreen", edgecolor="black")
plt.title("Exit Probability by Current Group"); plt.ylabel("P(exit)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"exit_probs.png")); plt.show()

# Optional: breakdown of "Other‑Small" -------------------------------
if "Other‑Small" in df["group"].unique():
    top_other = (df.loc[df["group"]=="Other‑Small","simple"]
                   .value_counts().head(10))
    plt.figure(figsize=(8,4))
    top_other.plot(kind="bar", color="grey", edgecolor="black")
    plt.title("Top constituents inside ‘Other‑Small’")
    plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"other_small_breakdown.png"))
    plt.show()
