"""
TRANSITIONS ANALYSIS SCRIPT
---------------------------
This script loads a hospital transfers dataset from "transfers.csv" and performs
the following:
  1. Data Loading & Cleaning:
     - Parse datetime fields ("intime", "outtime")
     - Compute event durations (in minutes)
     - Remove records with extremely short (<1 min) or very long durations (>30 days)
     - Extract additional time features (hour, date, week)
     
  2. Care Unit Label Simplification & Grouping:
     - Simplify individual care unit names using a custom mapping.
     - Further group the simplified care units into clinically meaningful domains
       (for example, "Emergency", "ICU", "Medical Ward", "Surgical", "Admin/Other").
       
  3. Transition Matrix Computation:
     - Compute transitions (raw counts and probabilities) between consecutive care
       units for each patient.
     - Build a grouped transition matrix (by domain) for clarity.
     
  4. Visualizations:
     - Raw ED Arrival Histogram (by hour)
     - Normalized ED Arrivals per hour (average per day)
     - Weekly Transfer Trends (line plot via Plotly)
     - Boxplot of Event Durations (using log scale and capping at 99th percentile)
     - Transition Matrix Heatmap (thresholded for significant probabilities)
     - Hierarchical Clustermap (for detailed inspection; optional)
     - Sankey Diagram (of grouped transitions)
     
This multi-level approach is designed to be both scientifically rigorous and visually readable.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# 1. DATA LOADING & PREPROCESSING˜
# -----------------------------
# Load transfers dataset; ensure your CSV is named "transfers.csv" and in the same folder.
df = pd.read_csv("unused/transfers.csv", parse_dates=["intime", "outtime"])

# Drop rows missing critical values (intime and eventtype)
df = df.dropna(subset=["intime", "eventtype"])

# Parse outtime and compute duration (in minutes)
df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
df["duration_minutes"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 60

# Filter out unrealistic durations:
# Keep durations >= 1 minute and <= 43200 minutes (i.e., 30 days)
df = df[(df["duration_minutes"] >= 1) & (df["duration_minutes"] <= 43200)]

# Extract additional time features
df["hour"] = df["intime"].dt.hour             # Hour of event
df["date"] = df["intime"].dt.date             # Date (without time)
df["week"] = df["intime"].dt.to_period("W").dt.start_time  # Week start

# -----------------------------
# 2. CARE UNIT SIMPLIFICATION & GROUPING
# -----------------------------
# Function to simplify individual care unit labels.
def simplify_careunit(unit):
    mapping = {
        "Emergency Department": "ED",
        "Emergency Department Observation": "ED Obs",
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro ICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "Cardiac ICU",
        "Intensive Care Unit (ICU)": "ICU",
        "Discharge Lounge": "Discharge",
        "Medicine/Cardiology": "Cardiology",
        "Coronary Care Unit (CCU)": "Cardiology",
        "Med/Surg": "MedSurg",
        "Med/Surg/Trauma": "MedSurg",
        "Med/Surg/GYN": "MedSurg",
        "Medical/Surgical (Gynecology)": "MedSurg",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "ICU",
        "Neurology": "Neuro",
        "Neuro Intermediate": "Neuro",
        "Neuro Stepdown": "Neuro",
        "Hematology/Oncology": "Oncology",
        "Hematology/Oncology Intermediate": "Oncology",
        "Surgery/Trauma": "Surgery",
        "Surgery": "Surgery",
        "Surgery/Vascular/Intermediate": "Surgery",
        "Thoracic Surgery": "Surgery",
        "Transplant": "Transplant",
        "Labor & Delivery": "OB",
        "Obstetrics Postpartum": "OB",
        "Obstetrics Antepartum": "OB",
        "Nursery": "Nursery",
        "Special Care Nursery (SCN)": "Nursery",
        "Psychiatry": "Psych",
        "Observation": "Obs",
        "UNKNOWN": "Unknown",
        "Unknown": "Unknown"
    }
    return mapping.get(unit, unit[:12])  # If not mapped, use the first 12 characters as a fallback

# Apply the simplification to create a new column.
df["careunit_simple"] = df["careunit"].apply(simplify_careunit)

# Additionally, group care units into broader domains for higher-level analysis.
def group_careunit(unit):
    # Use the simplified label for grouping logic.
    simple = simplify_careunit(unit)
    if simple in ["ED", "ED Obs"]:
        return "Emergency"
    elif simple in ["MICU", "SICU", "Neuro ICU", "ICU"]:
        return "ICU"
    elif simple in ["MedSurg", "Cardiology", "Oncology", "Medicine"]:
        return "Medical Ward"
    elif simple in ["Surgery", "Transplant", "OB"]:
        return "Surgical"
    elif simple in ["Discharge", "Obs", "Unknown"]:
        return "Admin"
    else:
        return "Other"

df["careunit_group"] = df["careunit"].apply(group_careunit)

# -----------------------------
# 3. TRANSITION MATRIX COMPUTATION
# -----------------------------
# Compute transitions between care units for each patient (individual-level)
df_sorted = df.sort_values(["subject_id", "intime"])
df_sorted["next_cu"] = df_sorted.groupby("subject_id")["careunit_simple"].shift(-1)
transitions_raw = df_sorted.dropna(subset=["careunit_simple", "next_cu"])
# Build raw transition count matrix
raw_matrix = transitions_raw.groupby(["careunit_simple", "next_cu"]).size().unstack().fillna(0)
raw_probs = raw_matrix.div(raw_matrix.sum(axis=1), axis=0)

# Compute transitions between grouped domains
df_sorted["next_group"] = df_sorted.groupby("subject_id")["careunit_group"].shift(-1)
transitions_group = df_sorted.dropna(subset=["careunit_group", "next_group"])
group_matrix = transitions_group.groupby(["careunit_group", "next_group"]).size().unstack().fillna(0)
group_probs = group_matrix.div(group_matrix.sum(axis=1), axis=0)

# -----------------------------
# 4. VISUALIZATIONS
# -----------------------------
# A. ED Arrivals Plots
def plot_ed_arrivals():
    # Raw ED arrivals per hour
    df_ed = df[df["eventtype"] == "ED"]
    plt.figure(figsize=(10, 4))
    sns.histplot(df_ed["hour"], bins=24, color="orange", edgecolor="black")
    plt.title("ED Arrivals per Hour (Raw Counts)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_normalized_ed_arrivals():
    # Normalized plot: average arrivals per hour per day.
    ed = df[df["eventtype"] == "ED"].copy()
    # Remove possible duplicates (same patient same time)
    ed = ed.sort_values(by=["subject_id", "intime"]).drop_duplicates(subset=["subject_id", "intime"])
    ed["date"] = ed["intime"].dt.date
    ed["hour"] = ed["intime"].dt.hour
    hourly = ed.groupby(["date", "hour"]).size().reset_index(name="count")
    mean_hourly = hourly.groupby("hour")["count"].mean()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=mean_hourly.index, y=mean_hourly.values, color="steelblue")
    plt.title("Average ED Arrivals per Hour (Normalized per Day)")
    plt.xlabel("Hour")
    plt.ylabel("Average Count")
    plt.xticks(range(24))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# B. Weekly Transfer Trends
def plot_weekly_transfers():
    weekly_counts = df.groupby(["week", "careunit_group"]).size().reset_index(name="count")
    # Using Plotly Express for an interactive line plot
    fig = px.line(weekly_counts, x="week", y="count", color="careunit_group",
                  title="Weekly Transfers per Grouped Domain")
    fig.show()

# C. Event Duration Boxplots
def plot_event_durations():
    
    # Suppose 'df' is your DataFrame with 'duration_minutes' and 'eventtype'
    df_linear = df[df['duration_minutes'] < df['duration_minutes'].quantile(0.99)]  # drop top 1% outliers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # (A) Linear-scale boxplot (99% capped)
    sns.boxplot(x="eventtype", y="duration_minutes", data=df_linear, ax=ax1)
    ax1.set_title("Duration per Event Type (Linear, 99% Cap)")
    ax1.set_yscale("linear")
    ax1.set_ylabel("Duration (minutes)")
    ax1.grid(True)

    # (B) Log-scale boxplot (full data)
    sns.boxplot(x="eventtype", y="duration_minutes", data=df, ax=ax2)
    ax2.set_title("Duration per Event Type (Log-Scaled, Full Range)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Duration (minutes, log scale)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



  

# D. Transition Matrix Visualizations
def plot_raw_transition_matrix(threshold=0.05):
    # Optionally apply a threshold to hide low-probability transitions.
    matrix = raw_probs.mask(raw_probs < threshold)
    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.title("Raw Transition Probability Matrix (Threshold ≥ 5%)")
    plt.xlabel("To Care Unit")
    plt.ylabel("From Care Unit")
    plt.tight_layout()
    plt.show()

def plot_grouped_transition_matrix(threshold=0.05):
    # Use grouped probabilities for a leaner view.
    matrix = group_probs.mask(group_probs < threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.title("Grouped Transition Probability Matrix (Threshold ≥ 5%)")
    plt.xlabel("To Group")
    plt.ylabel("From Group")
    plt.tight_layout()
    plt.show()

def plot_hierarchical_clustermap():
    # Hierarchical clustermap on the raw transition probabilities (for detailed pattern discovery)
    cg = sns.clustermap(raw_probs, cmap="Blues", figsize=(14, 14), annot=True, fmt=".2f")
    plt.title("Hierarchical Clustermap of Raw Transitions")
    plt.show()

def plot_sankey_diagram(threshold=0.05):
    # Build a Sankey diagram for grouped transitions.
    df_sankey = group_probs.stack().reset_index()
    df_sankey.columns = ["source", "target", "value"]
    df_sankey = df_sankey[df_sankey["value"] >= threshold]
    
    labels = list(pd.unique(df_sankey["source"].tolist() + df_sankey["target"].tolist()))
    label_index = {label: i for i, label in enumerate(labels)}
    
    # Build the links for the Sankey diagram.
    source_indices = df_sankey["source"].map(label_index)
    target_indices = df_sankey["target"].map(label_index)
    values = df_sankey["value"]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values
        )
    )])
    fig.update_layout(title_text="Sankey Diagram of Grouped Transitions (Threshold ≥ 5%)", font_size=10)
    fig.show()

# -----------------------------
# 5. EXECUTION: RUN ALL VISUALIZATIONS
# -----------------------------
if __name__ == "__main__":
    # Summary statistics for context
    print(f"Data timeframe: {df['intime'].min()} to {df['intime'].max()}")
    print(f"Total ED events: {len(df[df['eventtype'] == 'ED'])}")
    print(f"Number of unique simplified care units: {df['careunit_simple'].nunique()}")
    print(f"Number of unique grouped domains: {df['careunit_group'].nunique()}")

    # ED Arrivals
    plot_ed_arrivals()
    plot_normalized_ed_arrivals()
    
    # Weekly Transfer Trends (grouped)
    plot_weekly_transfers()
    
    # Event Duration Distribution
    plot_event_durations()
    
    # Transition Matrices
    plot_raw_transition_matrix()
    plot_grouped_transition_matrix()
    
    # (Optional) Hierarchical Clustermap for raw transitions
    # Uncomment the next line if you wish to see the clustermap.
    # plot_hierarchical_clustermap()
    
    # Sankey Diagram for Grouped Transitions
    plot_sankey_diagram()
