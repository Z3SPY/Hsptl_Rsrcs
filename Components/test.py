import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load the CSV files
# ---------------------------------------------------------
patients = pd.read_csv("F:/MMC/mimic-iv-2.1/hosp/patients.csv")
admissions = pd.read_csv("F:/MMC/mimic-iv-2.1/hosp/admissions.csv")


# ---------------------------------------------------------
# 2. Merge on subject_id so each admission gets its patient's anchor info
# ---------------------------------------------------------
df = pd.merge(admissions, patients, on="subject_id", how="inner")
df["admittime"] = pd.to_datetime(df["admittime"])

# ---------------------------------------------------------
# 3. Process by Anchor Year Group
# ---------------------------------------------------------
anchor_groups = df["anchor_year_group"].unique()

for group in anchor_groups:
    # Filter data to current anchor group
    df_group = df[df["anchor_year_group"] == group].copy()
    n_patients = df_group["subject_id"].nunique()
    print(f"Anchor Group: {group}, Number of Patients: {n_patients}")
    
    # -------------------------------
    # a) DAILY ADMISSION RATE
    # -------------------------------
    # Extract admission date (ignoring time)
    df_group["admission_date"] = df_group["admittime"].dt.date
    # Count admissions per day
    daily_counts = df_group.groupby("admission_date").size().reset_index(name="admissions")
    daily_counts["admission_date"] = pd.to_datetime(daily_counts["admission_date"])
    daily_counts = daily_counts.sort_values("admission_date")
    
    # Calculate daily rate: admissions per patient per day
    daily_counts["daily_rate"] = daily_counts["admissions"] / n_patients
    
    # Option 1: Weekly Aggregation
    # Convert each date to the starting date of its ISO week
    daily_counts["week"] = daily_counts["admission_date"].dt.to_period("W").dt.start_time
    weekly_counts = daily_counts.groupby("week").agg({"admissions": "sum"}).reset_index()
    weekly_counts["weekly_rate"] = weekly_counts["admissions"] / n_patients
    
    plt.figure(figsize=(12, 6))
    plt.bar(weekly_counts["week"], weekly_counts["weekly_rate"], width=5, edgecolor="black")
    plt.title(f"Weekly Admission Rate (per patient) for Anchor Group: {group}")
    plt.xlabel("Week (Deidentified)")
    plt.ylabel("Admission Rate (per patient per week)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Option 2: 7-Day Rolling Average on Daily Data
    daily_counts["rolling_rate"] = daily_counts["daily_rate"].rolling(window=7, min_periods=1).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_counts["admission_date"], daily_counts["rolling_rate"],
             marker="o", linestyle="-")
    plt.title(f"7-Day Rolling Average Daily Admission Rate (per patient)\nfor Anchor Group: {group}")
    plt.xlabel("Date (Deidentified)")
    plt.ylabel("Admission Rate (per patient per day)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # -------------------------------
    # b) HOURLY ADMISSION RATE
    # -------------------------------
    # Extract hour from admittime
    hourly_counts, hourly_bins = np.histogram(df_group["admittime"].dt.hour, bins=range(0, 25))
    hourly_rate = hourly_counts / n_patients  # Rate per patient per hour
    
    plt.figure(figsize=(8, 4))
    plt.bar(range(24), hourly_rate, width=0.8, edgecolor="black")
    plt.title(f"Hourly Admission Rate (per patient) for Anchor Group: {group}")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Admission Rate (per patient per hour)")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.show()
