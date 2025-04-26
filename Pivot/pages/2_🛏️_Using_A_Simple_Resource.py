'''
A Streamlit application based on the open treatment centre simulation model from Monks.T, Harper.A, Anagnoustou. A, Allen.M, Taylor.S. (2022)

Original Model: https://github.com/TomMonks/treatment-centre-sim/tree/main

Allows users to interact with an increasingly complex treatment simulation
'''
import asyncio
import gc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from helper_functions import add_logo, mermaid, center_running
from model_classes import Scenario, multiple_replications
from distribution_classes import Normal
from output_animation_functions import reshape_for_animations, generate_animation_df, generate_animation
# Set page parameters
st.set_page_config(
     page_title="Using a Simple Resource",
     layout="wide",
     initial_sidebar_state="expanded",
 )

# Add the logo
add_logo()

center_running()

# Import the stylesheet
with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("Discrete Event Simulation Playground")
st.subheader("Using a Simple Resource: Sending Patients to be Treated")

gc.collect()

# tab1, tab2, tab3 = st.tabs(["Introduction", "Exercise", "Playground"])
tab3, tab2, tab1 = st.tabs(["Information", "Exercise", "Playground"])

with tab3:

    st.markdown(
        """
        Now, it's all well and good having patients arrive, but at the moment there is no-one and nowhere to see them!

    We need to add our first resource.

    Resources exist inside our simulation, and can be nurses, rooms, ambulances - whatever we need them to be.

    When someone reaches the front of the queue, they will be allocated to a resource that is currently free.
    They will hold onto this resource for as long as they need it, and then they will let go of it and move on to the next part of the process.

    This means resources can continue to be reused again and again in the system, unlike our arrivals.

    So for now, let's make it so that when someone arrives, they need to be treated, and to do this they will need a resource.
    For now, we're keeping it simple - let's assume each nurse has a room that they treat people in. They always stay in this room, and as soon as a patient has finished being treated, the patient will leave and the nurse (and room) will become available again.

    This means we just have one type of resource to worry about.
    """
    )

    mermaid(height=175, code=
    """
            %%{ init: { 'flowchart': { 'curve': 'step' } } }%%
            %%{ init: {  'theme': 'base', 'themeVariables': {'lineColor': '#b4b4b4'} } }%%
            flowchart LR
            A[Arrival]----> B[Treatment]
            B -.-> C([Nurse/Cubicle\n<b>RESOURCE</b>])
            C -.-> B
            B ----> F[Discharge]

            classDef default font-size:18pt,font-family:lexend;
            linkStyle default stroke:white;
        """
    )

    st.markdown(
        """
        For now, we'll assume all of our patients are roughly equally injured - but there might still be some variation in how long it takes to treat them. Some might need a few stitches, some might just need a quick bit of advice.

        This time, we're going to sample from a different distribution - the normal distribution. A few people won't take very long to fix up, while a few might take quite a long time - but most of the people will take an amount of time that's somewhere in the middle.
        """)


    norm_dist_example = Normal(mean=50, sigma=10)
    norm_fig_example = px.histogram(norm_dist_example.sample(size=5000),
                                height=300)

    norm_fig_example.update_layout(yaxis_title="", xaxis_title="Consultation Time<br>(Minutes)")

    norm_fig_example.layout.update(showlegend=False,
                            margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(norm_fig_example, use_container_width=True)

    st.markdown("""
    We're going to start measuring a few more things now
    - how much of each resource's time is spent with patients **(known as resource utilisation)**
    - how long each patient waits before they get allocated a resource
    - what percentage of patients meet a target of being treated within 2 hours of turning up to our treatment centre
    """)




with tab2:
    st.markdown(
        """
        ### Things to Try Out

        - Try changing the sliders for consultation time and variation in consultation time.
        What happens to the shape of the graph below the sliders?
        ---
        - Put the consulation times back to the default (50 minutes length on average, 10 minutes of variation).
        Run the model and take a look at the animated flow of patients through the system. What do you notice about
            - the number of nurses in use? Do they ever get any breaks?
            - the size of the queue for treatment at different times - does it get bigger and smaller at different times, or just keep growing?
        ---
        - What happens when you play around with the number of nurses we have available?
            - Look at the queues, but look at the resource utilisation too. The resource utilisation tells us how much of the time each nurse is busy rather than waiting for a patient to turn up.
            - Can you find a middle ground where the nurse is being used a good amount without the queues building up?
        ---
        """)

    with st.expander("Click here for bonus exercises"):
        st.markdown(
        """
        - What happens to the average utilisation and waits when you keep the number of nurses the same but change
            - the average length of time it takes each patient to be seen?
            - the variability in the length of time it takes each patient to be seen?

        """
    )


with tab1:
    col1, col2 = st.columns(2)

    with col1:

        nurses = st.slider("👨‍⚕️👩‍⚕️ How Many Rooms/Nurses Are Available?", 1, 15, step=1, value=4)

        seed = st.slider("🎲 Set a random number for the computer to start from",
                            1, 1000,
                            step=1, value=42)

        with st.expander("Previous Parameters"):

            st.markdown("If you like, you can edit these parameters too!")

            n_reps = st.slider("🔁 How many times should the simulation run?",
                            1, 30,
                            step=1, value=6)

            run_time_days = st.slider("🗓️ How many days should we run the simulation for each time?",
                                    1, 40,
                                    step=1, value=10)


            mean_arrivals_per_day = st.slider("🧍 How many patients should arrive per day on average?",
                                            10, 300,
                                            step=5, value=120)

    with col2:

        consult_time = st.slider("⏱️ How long (in minutes) does a consultation take on average?",
                                    5, 150, step=5, value=50)

        consult_time_sd = st.slider("🕔 🕣 How much (in minutes) does the time for a consultation usually vary by?",
                                    5, 30, step=5, value=10)

        norm_dist = Normal(consult_time, consult_time_sd, random_seed=seed)
        norm_fig = px.histogram(norm_dist.sample(size=2500), height=150)

        norm_fig.update_layout(yaxis_title="", xaxis_title="Consultation Time<br>(Minutes)")

        norm_fig.update_xaxes(tick0=0, dtick=10, range=[0,
                                                        # max(norm_dist.sample(size=2500))
                                                        240
                                                        ])



        norm_fig.layout.update(showlegend=False,
                                margin=dict(l=0, r=0, t=0, b=0))

        st.markdown("#### Consultation Time Distribution")
        st.plotly_chart(norm_fig,
                        use_container_width=True,
                        config = {'displayModeBar': False})



    # A user must press a streamlit button to run the model
    button_run_pressed = st.button("Run simulation")


    if button_run_pressed:

        # add a spinner and then display success box
        with st.spinner('Simulating the minor injuries unit...'):
            args = Scenario(
                random_number_set=seed,
                n_cubicles_1=nurses,
                override_arrival_rate=True,
                manual_arrival_rate=60/(mean_arrivals_per_day/24),
                model="simplest",
                trauma_treat_mean=consult_time,
                trauma_treat_var=consult_time_sd
                )
            #await asyncio.sleep(0.1)
            # run multiple replications of experment
            detailed_outputs = multiple_replications(
                args,
                n_reps=n_reps,
                rc_period=run_time_days*60*24,
                return_detailed_logs=True
            )

            results = pd.concat([detailed_outputs[i]['results']['summary_df'].assign(rep= i+1)
                                        for i in range(n_reps)]).set_index('rep')

            full_event_log = pd.concat([detailed_outputs[i]['results']['full_event_log'].assign(rep= i+1)
                                        for i in range(n_reps)])

            del detailed_outputs
            gc.collect()

            animation_dfs_log = reshape_for_animations(
                        event_log=full_event_log[
                            (full_event_log['rep']==1) &
                            ((full_event_log['event_type']=='queue') | (full_event_log['event_type']=='resource_use')  | (full_event_log['event_type']=='arrival_departure'))
                            # Limit to first 5 days
                        ],
                        every_x_time_units=5,
                        step_snapshot_max=45,
                        limit_duration=60*24*5
                )

            del full_event_log
            gc.collect()

    if button_run_pressed:
        tab1, tab2, tab3 = st.tabs(
                ["Animated Log", "Simple Graphs", "Advanced Graphs"]
            )

    #     st.markdown("""
    # You can click on the three tabs below ("Animated Log", "Simple Graphs", and "Advanced Graphs") to view different outputs from the model.
    #                 """)


        with tab1:
            # st.write(results)
            st.subheader("Animated Model Output")
            with st.spinner('Generating the animated patient log...'):

                event_position_df = pd.DataFrame([
                            {'event': 'arrival', 'x':  50, 'y': 300, 'label': "Arrival" },

                            # Triage - minor and trauma
                            {'event': 'treatment_wait_begins', 'x':  200, 'y': 170, 'label': "Waiting for Treatment"  },
                            {'event': 'treatment_begins', 'x':  200, 'y': 110, 'resource':'n_cubicles_1', 'label': "Being Treated" },

                            {'event': 'exit',
                            'x':  270, 'y': 70, 'label': "Exit"}

                        ])

                st.markdown(
    """
    The plot below shows a snapshot every 5 minutes of the position of everyone in our emergency department model.

    The buttons to the left of the slider below the plot can be used to start and stop the animation.

    Clicking on the bar below the plot and dragging your cursor to the left or right allows you to rapidly jump through to a different time in the simulation.

    Only the first replication of the simulation is shown.
    """
                )

                full_patient_df_plus_pos = generate_animation_df(
                    full_patient_df=animation_dfs_log,
                    event_position_df = event_position_df,
                    wrap_queues_at=15,
                    gap_between_entities=10,
                    gap_between_rows=20,
                    step_snapshot_max=45
                    )

                st.plotly_chart(generate_animation(
                                    full_patient_df_plus_pos=full_patient_df_plus_pos,
                                    event_position_df = event_position_df,
                                    scenario=args,
                                    include_play_button=True,
                                    plotly_height=700,
                                    plotly_width=1200,
                                    override_x_max=300,
                                    override_y_max=500,
                                    time_display_units="dhm",
                                    display_stage_labels=False,
                                    add_background_image="https://raw.githubusercontent.com/hsma-programme/Teaching_DES_Concepts_Streamlit/main/resources/Simplest%20Model%20Background%20Image%20-%20Horizontal%20Layout.drawio.png",

                            ), use_container_width=False,
                               config = {'displayModeBar': False})

                del animation_dfs_log
                gc.collect()

        with tab2:
            in_range_util = sum((results.mean().filter(like="util")<0.85) & (results.mean().filter(like="util") > 0.65))
            in_range_wait = sum((results.mean().filter(like="wait")<120))

            col_res_a, col_res_b = st.columns([1,1])

            with col_res_a:
                st.metric(label=":bed: **Utilisation Metrics in Ideal Range**", value="{} of {}".format(in_range_util, len(results.mean().filter(like="util"))))

                #util_fig_simple = px.bar(results.mean().filter(like="util"), opacity=0.5)
                st.markdown(
                    """
                    The emergency department wants to aim for an average of 65% to 85% utilisation across all resources in the emergency department.
                    The green box shows this ideal range. If the bars overlap with the green box, utilisation is ideal.

                    If utilisation is below this, you might want to **reduce** the number of those resources available.

                    If utilisation is above this point, you may want to **increase** the number of that type of resource available.
                    """
                )
                util_fig_simple = go.Figure()
                # Add optimum range
                util_fig_simple.add_hrect(y0=0.65, y1=0.85,
                                          fillcolor="#5DFDA0", opacity=0.25,  line_width=0)
                # Add extreme range (above)
                util_fig_simple.add_hrect(y0=0.85, y1=1,
                                          fillcolor="#D45E5E", opacity=0.25, line_width=0)
                # Add suboptimum range (below)
                util_fig_simple.add_hrect(y0=0.4, y1=0.65,
                                          fillcolor="#FDD049", opacity=0.25, line_width=0)
                # Add extreme range (below)
                util_fig_simple.add_hrect(y0=0, y1=0.4,
                                          fillcolor="#D45E5E", opacity=0.25, line_width=0)

                util_fig_simple.add_bar(x=results.mean().filter(like="util").index.tolist(),
                                        y=results.mean().filter(like="util").tolist())

                util_fig_simple.update_layout(yaxis_tickformat = '.3%')
                util_fig_simple.update_yaxes(title_text='Resource Utilisation (%)',
                                             range=[-0.05, 1.1])
                # util_fig_simple.data = util_fig_simple.data[::-1]
                util_fig_simple.update_xaxes(labelalias={
                    "01b_treatment_util": "Treatment Bays",
                }, tickangle=0)

                util_fig_simple.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis_tickformat = '.0%',
                    title=dict(text="Utilisation of Resources - Average Across Simulation Runs",
                               automargin=True,
                               yref='paper')
                    )

                st.plotly_chart(
                    util_fig_simple,
                    use_container_width=True,
                    config = {'displayModeBar': False}
                )


            with col_res_b:
                #util_fig_simple = px.bar(results.mean().filter(like="wait"), opacity=0.5)
                st.metric(label=":clock2: **Wait Metrics in Ideal Range**", value="{} of {}".format(in_range_wait, len(results.mean().filter(like="01b"))))

                st.markdown(
                    """
                    The emergency department wants to ensure people wait no longer than 2 hours (120 minutes) to be seen.
                    This needs to be balanced with the utilisation graphs on the left.

                    The green box shows waits of less than two hours. If the bars fall within this range, the number of resources does not need to be changed.
                    """
                )

                wait_fig_simple = go.Figure()
                wait_fig_simple.add_hrect(y0=0, y1=60*2, fillcolor="#5DFDA0",
                                          opacity=0.3, line_width=0)

                wait_fig_simple.add_bar(x=results.mean().filter(like="01a").index.tolist(),
                                        y=results.mean().filter(like="01a").tolist())

                wait_fig_simple.update_xaxes(labelalias={
                    "01a_treatment_wait": "Treatment"
                }, tickangle=0)
                # wait_fig_simple.data = wait_fig_simple.data[::-1]
                wait_fig_simple.update_yaxes(title_text='Wait for Treatment Stage (Minutes)')

                wait_fig_simple.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    title=dict(text="Waits at Each Step - Average Across Simulation Runs",
                               automargin=True,
                               yref='paper')
                    )

                st.plotly_chart(
                    wait_fig_simple,
                    use_container_width=True,
                    config = {'displayModeBar': False}
                )

            col_res_c, col_res_d = st.columns(2)

            with col_res_c:
                #util_fig_simple = px.bar(results.mean().filter(like="wait"), opacity=0.5)
                st.metric(label=":clock2: **Wait Target Met**", value="{} of {}".format(in_range_wait, len(results.mean().filter(like="01c"))))

                st.markdown(
                    """
                    The emergency department wants to ensure people wait no longer than 2 hours (120 minutes) to be seen.
                    This needs to be balanced with the utilisation graphs on the left.

                    The green box shows waits of less than two hours. If the bars fall within this range, the number of resources does not need to be changed.
                    """
                )

                wait_target_simple = go.Figure()
                wait_target_simple.add_hrect(y0=0.85, y1=1, fillcolor="#5DFDA0",
                                          opacity=0.3, line_width=0)

                wait_target_simple.add_bar(x=results.median().filter(like="01c").index.tolist(),
                                        y=results.median().filter(like="01c").tolist())

                wait_target_simple.update_xaxes(labelalias={
                    "01c_treatment_wait_target_met": "Treatment Wait - Target Met"
                }, tickangle=0)
                # wait_fig_simple.data = wait_fig_simple.data[::-1]
                wait_target_simple.update_yaxes(title_text='Average % of patients where 2 hour wait target met')

                wait_target_simple.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                                                 yaxis_tickformat = '.0%',
                    title=dict(text="% of Patients Waiting Less than Target Time - Average Across Simulation Runs",
                               automargin=True,
                               yref='paper')
                    )

                st.plotly_chart(
                    wait_target_simple,
                    use_container_width=True,
                    config = {'displayModeBar': False}
                )


                # st.write(results)
        with tab3:

            st.markdown(
            """
            We can use **box plots** to help us understand the variation in each result during a model run.

            Because of the variation in the patterns of arrivals, as well as the variation in the length of consultations, we may find that sometimes model runs fall within our desired ranges but other times, despite the parameters being the same, they don't.

            This gives us a better idea of how likely a redesigned system is to meet the targets.
            """
            )

            st.markdown("""
                        ### Utilisation
                        """)
            util_box = px.box(
                    results.reset_index().melt(id_vars=["rep"]).set_index('variable').filter(like="util", axis=0).reset_index(),
                    y="variable",
                    x="value",
                    points="all",
                    range_x=[0, 1.1],
                    height=200)

            util_box.update_layout(yaxis_title="",
                                   xaxis_title="Average Utilisation in Model Run",
                                   xaxis_tickformat = '.0%')

            util_box.add_vrect(x0=0.65, x1=0.85,
                                          fillcolor="#5DFDA0", opacity=0.25,  line_width=0)
            # Add extreme range (above)
            util_box.add_vrect(x0=0.85, x1=1,
                                        fillcolor="#D45E5E", opacity=0.25, line_width=0)
            # Add suboptimum range (below)
            util_box.add_vrect(x0=0.4, x1=0.65,
                                        fillcolor="#FDD049", opacity=0.25, line_width=0)
            # Add extreme range (below)
            util_box.add_vrect(x0=0, x1=0.4,
                                        fillcolor="#D45E5E", opacity=0.25, line_width=0)

            util_box.update_yaxes(labelalias={
                "01b_treatment_util": "Treatment<br>Bays"
            }, tickangle=0)



            st.plotly_chart(util_box,
                    use_container_width=True
                )


            st.markdown("""
                        ### Waits
                        """)
            wait_box = px.box(
                    results.reset_index().melt(id_vars=["rep"]).set_index('variable').filter(like="01a", axis=0).reset_index(),
                    y="variable",
                    x="value",
                    points="all",
                    height=200,
                    range_x=[0, results.reset_index().melt(id_vars=["rep"]).set_index('variable').filter(like="01a", axis=0).reset_index().max().value]
                    )
            wait_box.update_layout(yaxis_title="",
                                   xaxis_title="Average Wait in Model Run")

            wait_box.update_yaxes(labelalias={
                    "01a_treatment_wait": "Treatment Wait"
                }, tickangle=0)

            wait_box.add_vrect(x0=0, x1=60*2, fillcolor="#5DFDA0",
                                          opacity=0.3, line_width=0)

            st.plotly_chart(wait_box,
                    use_container_width=True
                )

            st.markdown("""
                        ### Wait Targets
                        This is the percentage of clients who met the 2 hour wait target.
                        """)

            wait_target_box = px.box(
                    results.reset_index().melt(id_vars=["rep"]).set_index('variable').filter(like="1c", axis=0).reset_index(),
                    y="variable",
                    x="value",
                    points="all",
                    height=200,
                    range_x=[0, 1.1]
                    )

            wait_target_box.update_layout(yaxis_title="",
                                          xaxis_title="% of clients meeting waiting time target",
                                          xaxis_tickformat = '.0%')

            wait_target_box.update_yaxes(labelalias={
                "01c_treatment_wait_target_met": "Waiting Time Target<br>(% met)"
            }, tickangle=0)


            st.plotly_chart(wait_target_box,
                    use_container_width=True
                )


            st.markdown("""
                        ### Throughput
                        This is the percentage of clients who entered the system who had left by the time the model stopped running.
                        Higher values are better - low values suggest a big backlog of people getting stuck in the system for a long time.

                        Note that this isn't a good metric to compare across different lengths of model run, but can be useful to consider for the same length of run with different parameters.
                        """)

            results['perc_throughput'] = results['09_throughput']/results['00_arrivals']
            throughput_box = px.box(
                    results.reset_index().melt(id_vars=["rep"]).set_index('variable').filter(like="perc_throughput", axis=0).reset_index(),
                    y="variable",
                    x="value",
                    points="all",
                    height=200,
                    range_x=[0, 1.1]
                    )

            throughput_box.update_layout(yaxis_title="",
                                         xaxis_title="% Throughput in Model Run",
                                         xaxis_tickformat = '.0%')

            throughput_box.update_yaxes(labelalias={
                "perc_throughput": "Throughput (%)"
            }, tickangle=0)


            st.plotly_chart(throughput_box,
                    use_container_width=True
                )

            del results
            gc.collect()

            # st.write(full_event_log)
