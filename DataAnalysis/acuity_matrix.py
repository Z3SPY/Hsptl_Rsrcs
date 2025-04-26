import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ───────────────────────────────────────────────────
# 1. Load your core CSV files
# ───────────────────────────────────────────────────
transfers = pd.read_csv("unused/transfers.csv/transfers.csv")
triage = pd.read_csv("unused/triage.csv/triage.csv")
edstays = pd.read_csv("unused/edstays.csv/edstays.csv")

# ───────────────────────────────────────────────────
# 2. Normalize careunit names and group them logically
# ───────────────────────────────────────────────────
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
        "Neuro Intermediate": "Neuro-Step",
        "Medical/Surgical (Gynecology)": "GYN-Step",
        "Transplant": "Tx Ward",
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

def group(unit: str) -> str:
    u = unit.lower()
    if u in ["ed", "ed obs"]:
        return "Emergency"
    if u in ["icu", "sicu", "micu", "neuro icu", "cardiac icu", "trauma sicu"]:
        return "ICU"
    if "step" in u:
        return "Step-Down"
    if u in ["medsurg", "medicine", "cardiology", "oncology", "tx ward", "neurology", "medicine/car"]:
        return "Medical Ward"
    if any(k in u for k in ["surg", "vascular", "surgery/panc"]):
        return "Surgical Ward"
    if any(k in u for k in ["ob ante", "ob post", "obstetrics", "l&d", "gyn"]):
        return "Obstetrics"
    if "nursery" in u or "scn" in u:
        return "Pediatrics"
    if "psych" in u:
        return "Psychiatry"
    if "pacu" in u:
        return "Recovery"
    if "obs" in u and "ed" not in u:
        return "Observation"
    if "discharge" in u or "unknown" in u:
        return "Discharge"
    return None

transfers["careunit"] = transfers["careunit"].map(simplify)
transfers["group"] = transfers["careunit"].map(group)
transfers = transfers.dropna(subset=["group", "hadm_id"])

# ───────────────────────────────────────────────────
# 3. Construct chronological transitions
# ───────────────────────────────────────────────────
transfers["intime"] = pd.to_datetime(transfers["intime"])
chron = transfers.sort_values(["subject_id", "hadm_id", "intime"]).copy()
chron["next_group"] = chron.groupby(["subject_id", "hadm_id"])["group"].shift(-1)
chron = chron.dropna(subset=["next_group"])

# ───────────────────────────────────────────────────
# 4. Merge in severity/acuity data from triage
# ───────────────────────────────────────────────────
triage = triage[["stay_id", "acuity"]].dropna()
triage["severity"] = triage["acuity"].astype(int)
edstays = edstays[["subject_id", "stay_id"]]

chron = chron.merge(edstays, on="subject_id", how="left")
chron = chron.merge(triage, on="stay_id", how="left")
chron = chron.dropna(subset=["severity"])
chron["severity"] = chron["severity"].astype(int)

# ───────────────────────────────────────────────────
# 5. Count transitions per severity
# ───────────────────────────────────────────────────
records = []
for sev, df in chron.groupby("severity"):
    M = df.groupby(["group", "next_group"]).size().unstack(fill_value=0)
    P = M.div(M.sum(axis=1), axis=0)
    for from_unit in P.index:
        if from_unit == "Discharge":
            continue
        for to_unit in P.columns:
            prob = P.loc[from_unit, to_unit]
            if prob > 0:
                records.append({
                    "severity": sev,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "probability": prob
                })

# ───────────────────────────────────────────────────
# 6. Save to CSV and show heatmap per severity
# ───────────────────────────────────────────────────
df_trans = pd.DataFrame(records)
os.makedirs("DataAnalysis/graph", exist_ok=True)
outpath = "DataAnalysis/graph/P_within_stay_per_acuity.csv"
df_trans.to_csv(outpath, index=False)
print(f"✅ Saved: {outpath}")

for sev in sorted(df_trans["severity"].unique()):
    heat = df_trans[df_trans["severity"] == sev]
    heat_pivot = heat.pivot(index="from_unit", columns="to_unit", values="probability")
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat_pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title(f"Transition Probabilities (Severity={sev})")
    plt.tight_layout()
    plt.savefig(f"DataAnalysis/graph/transition_heatmap_sev{sev}.png")
    plt.close()

print("✅ All severity heatmaps saved!")
