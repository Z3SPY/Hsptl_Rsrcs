import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# ---------------------------------------------------------
# 1. Load the CSV files
# ---------------------------------------------------------
patients = pd.read_csv("unused/patients.csv/patients.csv")
admissions = pd.read_csv("unused/admissions.csv/admissions.csv")


# ---------------------------------------------------------
# 2. Merge on subject_id so each admission gets its patient's anchor info
# ---------------------------------------------------------
df = pd.merge(admissions, patients, on="subject_id", how="inner")
df["admittime"] = pd.to_datetime(df["admittime"])

df.to_csv("test.csv", sep=',', index=False, encoding='utf-8')


def PlotTheData(cur_df, type):

    # -------------------------------
    # a) DAILY ADMISSION RATE
    # -------------------------------
    # Extract admission date (ignoring time)
    cur_df["admission_date"] = cur_df["admittime"].dt.date
    # Count admissions per day
    daily_counts = cur_df.groupby("admission_date").size().reset_index(name="admissions")
    daily_counts["admission_date"] = pd.to_datetime(daily_counts["admission_date"])
    daily_counts = daily_counts.sort_values("admission_date")
    
    # Calculate daily rate: admissions per patient per day
    daily_counts["daily_rate"] = daily_counts["admissions"] / n_patients
    
    # Weekly Aggregation and Visualization
    # Convert each date to the starting date of its ISO week
    daily_counts['week'] = daily_counts['admission_date'].dt.to_period("W").dt.start_time
    weekly_counts = daily_counts.groupby("week").agg({
        "admissions": "sum",
        "daily_rate": "mean"  # Calculate mean daily rate per week
    }).reset_index()
    
    # Create an interactive plot using plotly
    fig = go.Figure()
    
    # Add weekly admission bars
    fig.add_trace(go.Bar(
        x=weekly_counts["week"],
        y=weekly_counts["daily_rate"],
        name="Weekly Average Daily Rate",
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=weekly_counts["week"],
        y=weekly_counts["daily_rate"].rolling(window=4).mean(),
        name="4-Week Moving Average",
        line=dict(color='rgb(255, 127, 14)', width=3)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Weekly Admission Patterns for Anchor Group: {group}",
        xaxis_title="Week",
        yaxis_title="Average Daily Admission Rate",
        showlegend=True,
        template="plotly_white",
        height=500
    )
    
    # Show the plot
    fig.show()

    if type == "daily":
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

    elif type == "hourly":
        # -------------------------------
        # b) HOURLY ADMISSION RATE
        # -------------------------------
        # Extract hour from admittime
        hourly_counts, hourly_bins = np.histogram(cur_df["admittime"].dt.hour, bins=range(0, 25))
        hourly_rate = hourly_counts / n_patients  # Rate per patient per hour
        
        plt.figure(figsize=(8, 4))
        plt.bar(range(24), hourly_rate, width=0.8, edgecolor="black")
        plt.title(f"Hourly Admission Rate (per patient) for Anchor Group: {group}")
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("Admission Rate (per patient per hour)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.show()





# ---------------------------------------------------------
# 3. Process by Anchor Year Group
# ---------------------------------------------------------
anchor_groups = df["anchor_year_group"].unique()
run = True 


#"2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"
anchor_group_check = [ "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]




        
def plot_admission_histogram(df_group, group, normalize=False):
    df_group['admittime'] = pd.to_datetime(df_group['admittime'])
    df_group['week'] = df_group['admittime'].dt.isocalendar().week
    df_group['month'] = df_group['admittime'].dt.month
    df_group['year'] = df_group['admittime'].dt.year

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Weekly distribution
    weekly_counts = df_group.groupby(['year', 'week']).size().reset_index(name='counts')
    admission_distribution = weekly_counts['counts'].value_counts().sort_index()

    # Discrete histogram
    if normalize:
        ax1.bar(admission_distribution.index,
                admission_distribution.values / admission_distribution.sum(),
                color='skyblue', edgecolor='black')
        ax1.set_ylabel("Proportion of Weeks")
    else:
        ax1.bar(admission_distribution.index,
                admission_distribution.values,
                color='skyblue', edgecolor='black')
        ax1.set_ylabel("Number of Weeks with That Admission Count")

    ax1.set_title(f'Weekly Admission Count Distribution ({group})')
    ax1.set_xlabel('Number of Admissions in a Week')

    # Monthly boxplot
    monthly_counts = df_group.groupby(['year', 'month']).size().reset_index(name='counts')
    monthly_data = [monthly_counts[monthly_counts['month'] == m]['counts'].values for m in range(1, 13)]
    ax2.boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.set_title(f'Monthly Admission Variation ({group})')
    ax2.set_ylabel('Admissions per Month')

    plt.tight_layout()
    plt.show()


# Update the main code section to include the new visualization
if (run is True):
    for group in anchor_groups:
        df_group = df[df["anchor_year_group"] == group].copy()
        n_patients = df_group["subject_id"].nunique()
        print(f"Anchor Group: {group}, Number of Patients: {n_patients}")
        if group in anchor_group_check:
            print(f"Plotting data for group {group}...")
            PlotTheData(df_group, "")
            plot_admission_histogram(df_group, group)
        else:
            print(f"Skipping group {group} for testing purposes.")