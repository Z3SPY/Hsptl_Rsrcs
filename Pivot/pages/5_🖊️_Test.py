import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
import numpy as np
import pandas as pd
import streamlit as st

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hospital_env import HospitalSimEnv, DictToBoxAction
from model_classes import Scenario
from output_animation_functions import (
    reshape_for_animations,
    generate_animation_df,
    generate_animation,
)

# ─── Cache & load RL components ───────────────────────────────────────────────
@st.cache_resource
def load_rl_components():
    model = PPO.load("ppo_hospital_final.zip")
    dummy_env = DummyVecEnv([lambda: DictToBoxAction(
        HospitalSimEnv(Scenario(), rc_period=24*60, inject_resources=True)
    )])
    vec_norm = VecNormalize.load("vecnormalize_stats.pkl", dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    return model, vec_norm

# ─── Roll out one PPO-driven episode ──────────────────────────────────────────
def rollout_rl_episode(model, vec_norm, scenario, run_days):
    raw = HospitalSimEnv(scenario, rc_period=run_days*24*60, inject_resources=True)
    raw.seed(0)  # fix variability in base staffing
    wrapped = DictToBoxAction(raw)

    # apply normalization
    vec = VecNormalize.load("vecnormalize_stats.pkl", DummyVecEnv([lambda: wrapped]))
    vec.training = False
    vec.norm_reward = False

    obs, done = vec.reset(), False
    actions = []
    while not done:
        a_vec, _ = model.predict(obs, deterministic=True)
        actions.append(a_vec)
        obs, _, done, _ = vec.step(a_vec)

    wrapped.actions = actions
    return wrapped

# ─── Build Plotly animation from event log ────────────────────────────────────
def build_animation_from_event_log(event_log: pd.DataFrame, scenario: Scenario, run_days: int):
    valid = ['arrival_departure','queue','resource_use','resource_use_end','exit']
    filtered = event_log[event_log['event_type'].isin(valid)]
    if filtered.empty:
        return None

    total_minutes = scenario.rc_period
    reshaped = reshape_for_animations(
        event_log=filtered,
        step_snapshot_max=30,
        every_x_time_units=5,
        limit_duration=total_minutes
    )

    anim_df = generate_animation_df(
        full_patient_df=reshaped,
        event_position_df=event_position_df,
        wrap_queues_at=10,
        gap_between_entities=10,
        gap_between_rows=25,
        step_snapshot_max=30
    )

    fig = generate_animation(
        full_patient_df_plus_pos=anim_df,
        event_position_df=event_position_df,
        scenario=scenario,
        include_play_button=True,
        plotly_height=800,
        plotly_width=1200,
        override_x_max=800,
        override_y_max=650,
        icon_and_text_size=16,
        display_stage_labels=False,
        time_display_units="dhm"
    )
    return fig

# ─── Pre-defined positions for each event ─────────────────────────────────────
event_position_df = pd.DataFrame([
    {'event':'triage_wait_begins','x':160,'y':400,'label':'Waiting for Triage'},
    {'event':'triage_begins','x':160,'y':315,'resource':'n_triage','label':'Being Triaged'},
    {'event':'MINORS_registration_wait_begins','x':290,'y':145,'label':'Waiting for Registration'},
    {'event':'MINORS_registration_begins','x':290,'y':85,'resource':'n_reg','label':'Being Registered'},
    {'event':'MINORS_examination_wait_begins','x':460,'y':145,'label':'Waiting for Examination'},
    {'event':'MINORS_examination_begins','x':460,'y':85,'resource':'n_exam','label':'Being Examined'},
    {'event':'MINORS_treatment_wait_begins','x':625,'y':145,'label':'Waiting for Treatment'},
    {'event':'MINORS_treatment_begins','x':625,'y':85,'resource':'n_cubicles_1','label':'Being Treated'},
    {'event':'TRAUMA_stabilisation_wait_begins','x':290,'y':560,'label':'Waiting for Stabilisation'},
    {'event':'TRAUMA_stabilisation_begins','x':290,'y':500,'resource':'n_trauma','label':'Being Stabilised'},
    {'event':'TRAUMA_treatment_wait_begins','x':625,'y':560,'label':'Waiting for Treatment'},
    {'event':'TRAUMA_treatment_begins','x':625,'y':500,'resource':'n_cubicles_2','label':'Being Treated'},
    {'event':'ward_admission','x':760,'y':145,'resource': 'n_ward_beds', 'label':'Admitted to Ward'},
    {'event':'ward_discharge','x':760,'y':85,'label':'Leaving Ward'},
    {'event':'icu_admission','x':760,'y':560,'resource': 'n_icu_beds', 'label':'Admitted to ICU'},
    {'event':'icu_discharge','x':760,'y':500,'label':'Leaving ICU'},
    {'event':'exit','x':670,'y':330,'label':'Exit'}
])

# ─── Streamlit page setup ────────────────────────────────────────────────────
st.set_page_config(page_title="ED Simulation", layout="wide")
st.title("🏥 ED Simulation: Manual vs. RL PPO")

# ─── Sidebar controls ────────────────────────────────────────────────────────
run_days = st.sidebar.slider("Simulation Length (Days)", 1, 10, 3)
capacities = {}
for key, label, default in [
    ("n_triage","Triage Bays",3),
    ("n_reg","Registration Clerks",2),
    ("n_exam","Exam Rooms",3),
    ("n_trauma","Stabilisation Bays",2),
    ("n_cub1","Cubicles (Minors)",3),
    ("n_cub2","Cubicles (Trauma)",2),
    ("n_ward","Ward Beds",10),
    ("n_icu","ICU Beds",5),
]:
    capacities[key] = st.sidebar.slider(label, 1, 20, default)

arrival_mode = st.sidebar.selectbox("Arrival Pattern", ["Empirical","Fixed Rate"])
if arrival_mode == "Fixed Rate":
    capacities['manual_arrival_rate'] = st.sidebar.slider("Patients per hour", 1, 20, 5)

# ─── Build Scenario ─────────────────────────────────────────────────────────
scenario = Scenario()
scenario.init_resource_counts(
    n_triage=capacities['n_triage'],
    n_reg=capacities['n_reg'],
    n_exam=capacities['n_exam'],
    n_trauma=capacities['n_trauma'],
    n_cubicles_1=capacities['n_cub1'],
    n_cubicles_2=capacities['n_cub2'],
    n_ward=capacities['n_ward'],
    n_icu=capacities['n_icu']
)
scenario.rc_period = run_days * 24 * 60
if arrival_mode == "Fixed Rate":
    scenario.override_arrival_rate = True
    scenario.manual_arrival_rate = capacities['manual_arrival_rate']
else:
    scenario.override_arrival_rate = False

# ─── Tabs: Manual vs. RL ────────────────────────────────────────────────────
tab_manual, tab_rl = st.tabs(["Manual Simulation", "RL PPO Simulation"])

# — Manual Simulation Tab
with tab_manual:
    st.header("🚶‍♂️ Manual MSO Simulation")
    if st.button("▶️ Run Manual Simulation"):
        with st.spinner("Running manual simulation…"):
            # Use a zero-action policy for Manual
            env = DictToBoxAction(
                HospitalSimEnv(scenario, rc_period=run_days*24*60, inject_resources=True)
            )
            obs, done = env.reset(), False
            zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
            while not done:
                obs, _, done, _ = env.step(zero)
            sim = env.env  # underlying SimPy model wrapper
        st.success("Manual simulation complete.")

        df_log = pd.DataFrame(sim.model.full_event_log)
        st.subheader("Event Log (Manual)")
        st.dataframe(df_log)

        fig = build_animation_from_event_log(df_log, scenario, run_days)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No events to animate.")

# — RL PPO Simulation Tab
with tab_rl:
    st.header("🤖 RL PPO-Driven Simulation")
    if st.button("▶️ Run RL PPO Simulation"):
        with st.spinner("Running RL PPO episode…"):
            model, vec_norm = load_rl_components()
            sim_env = rollout_rl_episode(model, vec_norm, scenario, run_days)
        st.success("RL episode complete.")

        # display event log
        df_log = pd.DataFrame(sim_env.model.full_event_log)
        st.subheader("Event Log (RL)")
        st.dataframe(df_log)

        # display actions
        if hasattr(sim_env, "actions") and sim_env.actions:
            actions_arr = np.vstack(sim_env.actions)
            cols = [unit for (unit, _) in sim_env.env.resource_units] + ["push","deferral"]
            df_act = pd.DataFrame(actions_arr, columns=cols)
            st.subheader("Actions (RL)")
            st.dataframe(df_act)

        # animate
        fig = build_animation_from_event_log(df_log, scenario, run_days)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No events to animate.")
