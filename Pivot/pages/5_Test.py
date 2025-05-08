
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from helper_functions import add_logo, mermaid, center_running
from model_classes import Scenario, multiple_replications, TreatmentCentreModel
from output_animation_functions import reshape_for_animations, generate_animation_df, generate_animation
from stable_baselines3 import PPO
from hospital_env import HospitalSimEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def run_baseline_simulation(_scenario, run_time_days):
    model = TreatmentCentreModel(args=_scenario)
    model.run(results_collection_period=60 * 24 * run_time_days)
    return model






event_position_df = pd.DataFrame([
         # Triage - minor and trauma
                {'event': 'triage_wait_begins',
                 'x':  160, 'y': 400, 'label': "Waiting for<br>Triage"  },
                {'event': 'triage_begins',
                 'x':  160, 'y': 315, 'resource':'n_triage', 'label': "Being Triaged" },

                # Minors (non-trauma) pathway
                {'event': 'MINORS_registration_wait_begins',
                 'x':  290, 'y': 145, 'label': "Waiting for<br>Registration"  },
                {'event': 'MINORS_registration_begins',
                 'x':  290, 'y': 85, 'resource':'n_reg', 'label':'Being<br>Registered'  },

                {'event': 'MINORS_examination_wait_begins',
                 'x':  460, 'y': 145, 'label': "Waiting for<br>Examination"  },
                {'event': 'MINORS_examination_begins',
                 'x':  460, 'y': 85, 'resource':'n_exam', 'label': "Being<br>Examined" },

                {'event': 'MINORS_treatment_wait_begins',
                 'x':  625, 'y': 145, 'label': "Waiting for<br>Treatment"  },
                {'event': 'MINORS_treatment_begins',
                 'x':  625, 'y': 85, 'resource':'n_cubicles_1', 'label': "Being<br>Treated" },

                # Trauma pathway
                {'event': 'TRAUMA_stabilisation_wait_begins',
                 'x': 290, 'y': 560, 'label': "Waiting for<br>Stabilisation" },
                {'event': 'TRAUMA_stabilisation_begins',
                 'x': 290, 'y': 500, 'resource':'n_trauma', 'label': "Being<br>Stabilised" },

                {'event': 'TRAUMA_treatment_wait_begins',
                 'x': 625, 'y': 560, 'label': "Waiting for<br>Treatment" },
                {'event': 'TRAUMA_treatment_begins',
                 'x': 625, 'y': 500, 'resource':'n_cubicles_2', 'label': "Being<br>Treated" },


                # Post-treatment ward/ICU pathways (shared by trauma and non-trauma)
                {'event': 'ward_admission',
                 'x': 760, 'y': 145, 'label': "Admitted to<br>Ward" },
                {'event': 'ward_discharge',
                 'x': 760, 'y': 85, 'label': "Leaving<br>Ward" },

                {'event': 'icu_admission',
                 'x': 760, 'y': 560, 'label': "Admitted to<br>ICU" },
                {'event': 'icu_discharge',
                 'x': 760, 'y': 500, 'label': "Leaving<br>ICU" },

                # Optional: Waiting queues before ICU/ward (only if modeled)
                {'event': 'TRAUMA_ward_wait_begins',
                 'x': 720, 'y': 145, 'label': "Waiting for<br>Ward Bed" },
                {'event': 'MINORS_ward_wait_begins',
                 'x': 720, 'y': 145, 'label': "Waiting for<br>Ward Bed" },

                {'event': 'TRAUMA_icu_wait_begins',
                 'x': 720, 'y': 560, 'label': "Waiting for<br>ICU Bed" },
                {'event': 'MINORS_icu_wait_begins',
                 'x': 720, 'y': 560, 'label': "Waiting for<br>ICU Bed" },



                 {'event': 'exit',
                 'x':  670, 'y': 330, 'label': "Exit"}
    ])



def build_animation_from_event_log(event_log, scenario, run_time_days):
    

    anim_df = generate_animation_df(
        full_patient_df=reshape_for_animations(
            event_log=event_log[
                event_log['event_type'].isin(['queue', 'resource_use', 'arrival_departure'])
            ],
            step_snapshot_max=30,
            every_x_time_units=5,
            limit_duration=60 * 24 * run_time_days
        ),
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
        plotly_height=900,
        plotly_width=1600,
        override_x_max=800,
        override_y_max=675,
        icon_and_text_size=19,
        display_stage_labels=False,
        time_display_units="dhm",
        add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Full%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
    )

    return fig


def simulate_with_trained_model(model_path, scenario, rc_period, use_normalizer=True, normalizer_path=None):

    env = HospitalSimEnv(sim_config=scenario, rc_period=rc_period)

    if use_normalizer and normalizer_path:
        env = VecNormalize.load(normalizer_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    return env.model  # model contains .full_event_log and summary_frame()


st.set_page_config(
     page_title="Model Train",
     layout="wide",
     initial_sidebar_state="expanded",
 )

# Initialise session state
if 'session_results' not in st.session_state:
    st.session_state['session_results'] = []

add_logo()

center_running()

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

## We add in a title for our web app's page
st.title("Discrete Event Simulation Playground")
st.subheader("How can we optimise the full system?")

st.markdown("Once you have run more than one scenario, try out the new tab 'compare scenario outputs'.")

gc.collect()

# tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Exercises", "Playground", "Compare Scenario Outputs"])
tab1, tab2 = st.tabs(["Manual MSO Simulation", "RL PPO Simulation"])



with tab1:
    st.header("🏥 Manual Simulation with MSO Logic")

    st.subheader("🛠️ Configure Simulation")
    run_time_days = st.slider("Simulation Length (Days)", 1, 10, value=3)

    st.markdown("### 💼 Staff and Resource Capacities")
    n_triage = st.slider("Triage Bays", 1, 10, 3)
    n_reg = st.slider("Registration Clerks", 1, 10, 2)
    n_exam = st.slider("Examination Rooms", 1, 10, 3)
    n_trauma = st.slider("Stabilisation Bays", 1, 10, 2)
    n_cub1 = st.slider("Cubicle 1 (Minors)", 1, 10, 3)
    n_cub2 = st.slider("Cubicle 2 (Trauma)", 1, 10, 2)
    n_ward = st.slider("Ward Beds", 1, 15, 5)
    n_icu = st.slider("ICU Beds", 1, 10, 3)

    st.markdown("### 🚶‍♂️ Arrival Pattern")
    arrival_mode = st.selectbox("Arrival Pattern", ["Empirical (Default)", "Fixed Rate"])
    manual_arrival_rate = 5
    if arrival_mode == "Fixed Rate":
        manual_arrival_rate = st.slider("Manual Arrival Rate (patients/hr)", 1, 20, 5)

    # ── Build Scenario Object ─────────────────────────────
    args = Scenario()
    args.init_resource_counts(
        n_triage=n_triage,
        n_reg=n_reg,
        n_exam=n_exam,
        n_trauma=n_trauma,
        n_cubicles_1=n_cub1,
        n_cubicles_2=n_cub2,
        n_ward=n_ward,
        n_icu=n_icu
    )
    if arrival_mode == "Fixed Rate":
        args.override_arrival_rate = True
        args.manual_arrival_rate = manual_arrival_rate
    else:
        args.override_arrival_rate = False

    # ── Run Simulation ────────────────────────────────────
    if st.button("▶️ Run Manual Simulation"):
        model = run_baseline_simulation(args, run_time_days)
        event_log = pd.DataFrame(model.full_event_log)

        fig = build_animation_from_event_log(event_log, args, run_time_days)
        st.success("✅ Simulation complete.")
        st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})




    
                
with tab2:
    st.header("🤖 Simulation Using Trained PPO Agent")

    st.subheader("🛠️ Configure PPO Simulation")
    run_time_days = st.slider("Simulation Length (Days)", 1, 10, value=3, key="rl_runtime")

    st.markdown("### 💼 Resource Capacities")
    n_triage = st.slider("Triage Bays", 1, 10, 3, key="rl_triage")
    n_reg = st.slider("Registration Clerks", 1, 10, 2, key="rl_reg")
    n_exam = st.slider("Examination Rooms", 1, 10, 3, key="rl_exam")
    n_trauma = st.slider("Stabilisation Bays", 1, 10, 2, key="rl_trauma")
    n_cub1 = st.slider("Cubicle 1 (Minors)", 1, 10, 3, key="rl_cub1")
    n_cub2 = st.slider("Cubicle 2 (Trauma)", 1, 10, 2, key="rl_cub2")
    n_ward = st.slider("Ward Beds", 1, 15, 5, key="rl_ward")
    n_icu = st.slider("ICU Beds", 1, 10, 3, key="rl_icu")

    st.markdown("### 🚶‍♂️ Arrival Pattern")
    arrival_mode_rl = st.selectbox("Arrival Pattern", ["Empirical (Default)", "Fixed Rate"], key="rl_arrival_mode")
    if arrival_mode_rl == "Fixed Rate":
        manual_arrival_rate_rl = st.slider("Manual Arrival Rate (patients/hr)", 1, 20, 5, key="rl_rate")

    multi_rep_eval = st.toggle("📊 Run Multiple Evaluations (no animation)", value=False)

    # ── Scenario Setup ──
    args = Scenario()
    args.init_resource_counts(
        n_triage=n_triage, n_reg=n_reg, n_exam=n_exam, n_trauma=n_trauma,
        n_cubicles_1=n_cub1, n_cubicles_2=n_cub2, n_ward=n_ward, n_icu=n_icu
    )
    if arrival_mode_rl == "Fixed Rate":
        args.override_arrival_rate = True
        args.manual_arrival_rate = manual_arrival_rate_rl
    else:
        args.override_arrival_rate = False

    # ── Run Simulation ──
    if st.button("▶️ Run PPO Simulation"):
        with st.spinner("Running PPO agent..."):

            model = PPO.load("ppo_hospital_final.zip")
            episode_logs = []
            summary_list = []
            num_episodes = 10 if multi_rep_eval else 1

            for i in range(num_episodes):
                # ✅ Create and wrap the env
                env = DummyVecEnv([
                    lambda: HospitalSimEnv(sim_config=args, rc_period=run_time_days * 60 * 24)
                ])
                
                # ✅ Load normalization stats ONCE
                env = VecNormalize.load("vecnormalize_stats.pkl", env)
                env.training = False
                env.norm_reward = False

                # ✅ Get the true environment inside
                base_env = env.envs[0]

                # ✅ Set arrival config if needed
                if arrival_mode_rl == "Empirical (Default)":
                    base_env.scenario.override_arrival_rate = False

                # ✅ Start rollout
                obs = env.reset()
                done = [False]
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)

                # ✅ Collect logs from the base env
                summary_list.append(base_env.model.sim_summary.summary_frame())  # ✅ correct
                episode_logs.extend(base_env.model.full_event_log)


        if multi_rep_eval:
            all_df = pd.concat(summary_list)
            st.success(f"✅ Completed {num_episodes} PPO evaluations.")
            st.dataframe(
                all_df.mean(numeric_only=True).to_frame(name="Average")
            )        
        else:
            st.success("✅ PPO Simulation Complete (1 episode).")

            # Defensive filtering: ensure 'event_type' exists
            event_log_df = pd.DataFrame(episode_logs)

            if 'event_type' in event_log_df.columns and 'time' in event_log_df.columns:
                filtered_log = event_log_df.query(
                    "event_type in ['queue', 'resource_use', 'arrival_departure']"
                )
            else:
                st.warning("⚠️ Skipping animation: missing 'event_type' or 'time' column in logs.")
                filtered_log = pd.DataFrame([])

            if not filtered_log.empty:
                anim_df = generate_animation_df(
                    full_patient_df=reshape_for_animations(
                        event_log=filtered_log,
                        step_snapshot_max=30,
                        every_x_time_units=5,
                        limit_duration=60 * 24 * run_time_days
                    ),
                    event_position_df=event_position_df,
                    wrap_queues_at=10,
                    gap_between_entities=10,
                    gap_between_rows=25,
                    step_snapshot_max=30
                )

                fig = generate_animation(
                    full_patient_df_plus_pos=anim_df,
                    event_position_df=event_position_df,
                    scenario=args,
                    include_play_button=True,
                    plotly_height=900,
                    plotly_width=1800,
                    override_x_max=700,
                    override_y_max=675,
                    icon_and_text_size=19,
                    display_stage_labels=False,
                    time_display_units="dhm",
                    add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Full%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",
                )

                st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})
            else:
                st.warning("⚠️ No animation generated — filtered event log is empty.")
