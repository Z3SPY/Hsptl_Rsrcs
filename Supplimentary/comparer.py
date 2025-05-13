# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Averaged RL Logs", layout="wide")
st.title("🏥 Averaged RL Training Logs")

@st.cache_data
def load_csvs(files):
    dfs = [pd.read_csv(f) for f in files]
    # concat & average by iteration
    avg = pd.concat(dfs).groupby("iter", as_index=False).mean()
    return avg

# upload
csvs = st.sidebar.file_uploader(
    "Upload one or more CSV log files",
    type="csv",
    accept_multiple_files=True
)

if not csvs:
    st.info("🔍 Upload your training-log CSVs on the left to compute averages.")
    st.stop()

# compute average
avg_df = load_csvs(csvs)

# show averaged dataframe & stats
st.header("📊 Averaged Data")
st.dataframe(avg_df)

st.subheader("Summary Statistics")
st.dataframe(avg_df.describe())

# trend plots
st.header("📈 Trends (Averaged)")

# 1) Return & CVaR
fig1 = px.line(
    avg_df,
    x="iter",
    y=["avg_return", "cvar"],
    labels={"value": "Metric", "variable": "Metric"},
    title="Average Return & CVaR"
)
st.plotly_chart(fig1, use_container_width=True)

# 2) Losses
fig2 = px.line(
    avg_df,
    x="iter",
    y=["policy_loss", "value_loss", "avg_loss"],
    title="Policy / Value / Avg Loss"
)
st.plotly_chart(fig2, use_container_width=True)

# 3) Entropy
st.subheader("Entropy (Exploration)")
st.line_chart(avg_df.set_index("iter")["entropy"])

# 4) Throughput
st.subheader("Env Steps per Second")
st.line_chart(avg_df.set_index("iter")["steps_per_s"])

# 5) Time vs Steps
fig5 = px.line(
    avg_df,
    x="iter",
    y=["env_steps", "elapsed_time_s"],
    labels={"value": "Count / Seconds", "variable": "Metric"},
    title="Env Steps & Elapsed Time"
)
st.plotly_chart(fig5, use_container_width=True)
