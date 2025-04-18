import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Load ED stays and patient metadata
# ---------------------------------------------------------
ed = pd.read_csv("unused/edstays.csv/edstays.csv")
patients = pd.read_csv("unused/patients.csv/patients.csv")

# Parse dates
ed["intime"] = pd.to_datetime(ed["intime"], dayfirst=True, errors='coerce')
ed["outtime"] = pd.to_datetime(ed["outtime"], dayfirst=True, errors='coerce')
ed = ed.dropna(subset=["intime", "outtime"])

# Calculate ED duration in minutes
ed["ed_minutes"] = (ed["outtime"] - ed["intime"]).dt.total_seconds() / 60

# 🔥 Remove invalid durations (e.g., negative or zero values)
ed = ed[ed["ed_minutes"] > 0]

# Merge ED stays with patient anchor group info
df = pd.merge(ed, patients[["subject_id", "gender", "anchor_year_group", "anchor_age"]], on="subject_id", how="left")
df = df.dropna(subset=["anchor_year_group"])

# ---------------------------------------------------------
# 2. Plotting Functions
# ---------------------------------------------------------
def PlotEDStayDuration(cur_df, group):
    cur_df["ed_date"] = cur_df["intime"].dt.date
    daily_los = cur_df.groupby("ed_date")["ed_minutes"].mean().reset_index(name="avg_los")
    daily_los["ed_date"] = pd.to_datetime(daily_los["ed_date"])
    daily_los = daily_los.sort_values("ed_date")

    daily_los["week"] = daily_los["ed_date"].dt.to_period("W").dt.start_time
    weekly_los = daily_los.groupby("week")["avg_los"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_los["week"],
        y=weekly_los["avg_los"],
        name="Weekly Avg ED Stay (min)",
        marker_color='rgb(26, 118, 255)'
    ))

    fig.add_trace(go.Scatter(
        x=weekly_los["week"],
        y=weekly_los["avg_los"].rolling(4).mean(),
        name="4-Week Rolling Avg",
        line=dict(color='rgb(255, 127, 14)', width=3)
    ))

    fig.update_layout(
        title=f"ED Stay Duration Over Time (Anchor Group: {group})",
        xaxis_title="Week",
        yaxis_title="Average ED Duration (minutes)",
        showlegend=True,
        template="plotly_white",
        height=500
    )
    fig.show()

def plot_ed_los_histogram(df_group, group, hours_cap=200):
    df_group = df_group.copy()
    df_group["ed_hours"] = df_group["ed_minutes"] / 60
    filtered = df_group[df_group["ed_hours"] < hours_cap]

    plt.figure(figsize=(10, 6))
    plt.hist(filtered["ed_hours"], bins=50, color="teal", edgecolor="black")
    plt.title(f"ED Stay Duration Histogram (<{hours_cap} hrs) ({group})")
    plt.xlabel("ED Stay (hours)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"\n--- Summary Stats for {group} ---")
    print(f"Mean: {filtered['ed_hours'].mean():.2f} hrs")
    print(f"Median: {filtered['ed_hours'].median():.2f} hrs")
    print(f"Std Dev: {filtered['ed_hours'].std():.2f} hrs")
    print(f"Variance: {filtered['ed_hours'].var():.2f} hrs")
    print(f"Min: {filtered['ed_hours'].min():.2f} hrs")
    print(f"Max: {filtered['ed_hours'].max():.2f} hrs")
    print(f"95th Percentile: {np.percentile(filtered['ed_hours'], 95):.2f} hrs")

# ---------------------------------------------------------
# 3. Main execution: loop over anchor year groups
# ---------------------------------------------------------
anchor_groups = df["anchor_year_group"].dropna().unique()
anchor_group_check = anchor_groups  # or manually filter

for group in anchor_groups:
    df_group = df[df["anchor_year_group"] == group].copy()
    n_patients = df_group["subject_id"].nunique()
    print(f"Anchor Group: {group}, Number of Patients: {n_patients}")

    if group in anchor_group_check:
        PlotEDStayDuration(df_group, group)
        plot_ed_los_histogram(df_group, group)
    else:
        print(f"Skipping group {group}.")
