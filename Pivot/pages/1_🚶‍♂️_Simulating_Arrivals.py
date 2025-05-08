'''
A Streamlit application based on the open treatment centre simulation model from Monks.T, Harper.A, Anagnoustou. A, Allen.M, Taylor.S. (2022)

Original Model: https://github.com/TomMonks/treatment-centre-sim/tree/main

Allows users to interact with an increasingly complex treatment simulation
'''
import time
import asyncio
import datetime as dt
import gc
import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st

from helper_functions import add_logo, mermaid, center_running
from model_classes import Scenario, multiple_replications
from distribution_classes import Exponential

st.set_page_config(
    page_title="Simulating Arrivals",
    layout="wide",
    initial_sidebar_state="expanded",
)

add_logo()

center_running()

# try:
#     running_on_st_community = st.secrets["IS_ST_COMMUNITY"]
# except FileNotFoundError:
#     running_on_st_community = False

with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# We add in a title for our web app's page
st.title("DES Playground")
st.subheader("Simulating Patients Arriving at the Centre")

gc.collect()

tab2, tab1 = st.tabs(["Information",  "Playground"])

with tab2:

    st.markdown(
        "Let's start with just having some patients arriving into our treatment centre.")


    st.markdown(
    """
    To start with, we need to create some simulated patients who will turn up to our centre.

    To simulate patient arrivals, we will use the exponential distribution, which looks a bit like this.

    """
    )

    exp_dist = Exponential(mean=5)
    exp_fig_example = px.histogram(exp_dist.sample(size=5000),
                                 width=600, height=300)

    exp_fig_example.layout.update(showlegend=False,
                            margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(exp_fig_example, use_container_width=True)

    


with tab1:
    col1_1, col1_2= st.columns(2)
    # set number of resources
    with col1_1:
        seed = st.slider("🎲 Set a random number for the computer to start from",
                        1, 1000,
                        step=1, value=103)

    with col1_2:
        run_time_days = st.slider("🗓️ How many days should we run the simulation for each time?",
                                  1, 31,
                                  step=1, value=15)

        n_reps = st.slider("🔁 How many times should the simulation run?",
                           1, 25,
                           step=1, value=10)


    mean_arrivals_per_day = 30 # Choose a number for testing | Set to TRUE to check if working

    args = Scenario(random_number_set=seed, manual_arrival_rate=60/(mean_arrivals_per_day/24), override_arrival_rate=False)

    # SIMULATION RUNNER
    button_run_pressed = st.button("Run simulation")
    if button_run_pressed:

        with st.spinner('Simulating the minor injuries unit...'):
           
            # Get Multiple Replications of the environment
            detailed_outputs = multiple_replications(
                args,
                n_reps=n_reps,
                rc_period=run_time_days*60*24,
                return_detailed_logs=True

            )

            patient_log = pd.concat([detailed_outputs[i]['results']['full_event_log'].assign(Rep= i+1)
                            for i in range(n_reps)])

            results = pd.concat([detailed_outputs[i]['results']['summary_df'].assign(rep= i+1)
                                                for i in range(n_reps)]).set_index('rep')


            patient_log = patient_log.assign(model_day = (patient_log.time/24/60).pipe(np.floor)+1)
            patient_log = patient_log.assign(time_in_day= (patient_log.time - ((patient_log.model_day -1) * 24 * 60)).pipe(np.floor))
            # patient_log = patient_log.assign(time_in_day_ (patient_log.time_in_day/60).pipe(np.floor))
            patient_log['patient_full_id'] = patient_log['Rep'].astype(str) + '_' + patient_log['patient'].astype(str)
            patient_log['rank'] = patient_log['time_in_day'].rank(method='max')

        #st.success('Done!')

        st.subheader(
            "Difference between average daily patients generated across simulation runs"
            )

        st.markdown(
            """
            The graph below shows the variation in the number of patients generated per day (on average) in a single simulation run.
            This can help us understand how much variation we get between model runs when we don't change parameters, only the random seed.

            The height of each bar is relative to the **first** simulation run.

            A bar that is **positive** shows that **more** patients were generated on average per day in that simulation than in the first simulation.

            A bar that is **negative** shows that **fewer* patients were generated on average per day in that simulation than in the first simulation.
            """
        )

        results['00a_arrivals_difference'] = ((results[['00_arrivals']].iloc[0]['00_arrivals'].astype(int))/run_time_days) - (results['00_arrivals']/run_time_days)

        results["colour_00a"] = np.where(results['00a_arrivals_difference']<0, 'neg', 'pos')
        # st.write(results)

        run_diff_bar_fig = px.bar(results.reset_index(drop=False),
                                  x="rep", y='00a_arrivals_difference',
                                  color="colour_00a")

        run_diff_bar_fig.update_layout(
            yaxis_title="Difference in daily patients between first run and this run",
            xaxis_title="Simulation Run")


        run_diff_bar_fig.layout.update(showlegend=False)
        run_diff_bar_fig.update_xaxes(tick0=1, dtick=1)

        st.plotly_chart(
            run_diff_bar_fig,
            use_container_width=True
            )
        
        status_text_string = 'The first simulation generated a total of {} patients (an average of {} patients per day)'.format(
            results[['00_arrivals']].iloc[0]['00_arrivals'].astype(int),
            (results[['00_arrivals']].iloc[0]
            ['00_arrivals']/run_time_days).round(1)
        )
        status_text = st.text(status_text_string)

        for i in range(n_reps-1):

            time.sleep(0.5)
            new_rows = results[['00_arrivals']].iloc[[i+1]]


            status_text_string = 'Simulation {} generated a total of {} patients (an average of {} patients per day)'.format(
                    i+2,
                    new_rows.iloc[0]['00_arrivals'].astype(int),
                    (new_rows.iloc[0]['00_arrivals']/run_time_days).round(1)
                ) + "\n" + status_text_string
            
            # Update status text.
            status_text.text(status_text_string)

        col_a_1, col_a_2 = st.columns(2)

        with col_a_1:
            st.subheader(
                "Histogram: Total Patients per Run"
            )

            st.markdown(
                """
                This plot shows the variation in the **total** daily patients per run.

                The horizontal axis shows the range of patients generated within a simulation run.

                The height of the bar shows how many simulation runs had an output in that group.
                """
            )

            total_fig = px.histogram(
                        results[['00_arrivals']],
                        nbins=5
                        )
            total_fig.layout.update(showlegend=False)

            total_fig.update_layout(yaxis_title="Number of Simulation Runs",
                                    xaxis_title="Total Patients Generated in Run")

            st.plotly_chart(
                total_fig,
                use_container_width=True
            )

        with col_a_2:
            st.subheader(
                    "Histogram: Average Daily Patients per Run"
            )

            st.markdown(
                """
                This plot shows the variation in the **average** daily patients per run.

                The horizontal axis shows the range of patients generated within a simulation run

                The height of the bar shows how many simulation runs had an output in that group.
                """
            )

            daily_average_fig = px.histogram(
                    (results[['00_arrivals']]/run_time_days).round(1),
                        nbins=5
                    )
            daily_average_fig.layout.update(showlegend=False)

            daily_average_fig.update_layout(yaxis_title="Number of Simulation Runs",
                                    xaxis_title="Average Daily Patients Generated in Run")

            st.plotly_chart(
                daily_average_fig,
                use_container_width=True
            )




        #facet_col_wrap_calculated = np.ceil(run_time_days/4).astype(int)

        patient_log['minute'] = dt.date.today() + pd.DateOffset(days=165) +  pd.TimedeltaIndex(patient_log['time'], unit='m')
        # https://strftime.org/
        patient_log['minute_display'] = patient_log['minute'].apply(lambda x: dt.datetime.strftime(x, '%d %B %Y\n%H:%M'))
        patient_log['minute_in_day'] = patient_log['minute'].apply(lambda x: dt.datetime.strftime(x, '%H:%M'))
        # patient_log['minute'] = patient_log['minute'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M'))


        tab1a, tab2a, tab3a = st.tabs(["Arrival plots by day", "Arrival plots by simulation run", "Cumulative Arrivals"])

        with tab1a:
            st.markdown(
            """
            The plots below show the minute-by-minute arrivals of patients across different model replications and different days.
            Only the first 10 replications and the first 5 days of the model are shown.

            Each dot is a single patient arriving.

            From left to right within each plot, we start at one minute past midnight and move through the day until midnight.

            Looking from the top to the bottom of each plot, we have the model replications.

            Each horizontal line of dots represents a **full day** for one model replication.

            Hovering over the dots will show the exact time that each patient arrived during that model replication and the number of patients that have arrived at that point in the simulation run.
            """
            )
            for i in range(5):
                st.markdown("### Day {}".format(i+1))

                minimal_log = patient_log[(patient_log['event'] == 'arrival') &
                                        (patient_log['Rep'] <= 10) &
                                        (patient_log['model_day'] == i+1)]

                minimal_log['Rep_str'] = minimal_log['Rep'].astype(str)

                time_plot = px.scatter(
                        minimal_log.sort_values("minute"),
                        x="minute",
                        y="Rep",
                        color="Rep_str",
                        custom_data=["Rep", "minute_in_day", "patient"],
                        category_orders={'Rep_str': [str(i+1) for i in range(10)]},
                        range_y=[0.5, min(10, n_reps)+0.5],
                        width=1200,
                        height=300,
                        opacity=0.5
                )

                del minimal_log

                time_plot.update_traces(
                    hovertemplate="<br>".join([
                        "Replication:%{customdata[0]}",
                        "Time of patient arrival: %{customdata[1]}",
                        "Arrival in this simulation run: %{customdata[2]}"
                    ])
                )

                time_plot.update_layout(yaxis_title="Simulation Run (Replication)",
                                        xaxis_title="Time",
                            yaxis = dict(
                            tickmode = 'linear',
                            tick0 = 1,
                            dtick = 1
                        ))

                time_plot.layout.update(showlegend=False,
                                        margin=dict(l=0, r=0, t=0, b=0))

                st.plotly_chart(time_plot, use_container_width=True)

            del time_plot
            gc.collect()

        with tab2a:
            st.markdown(
            """
            The plots below show the minute-by-minute arrivals of patients across different model replications and different days.
            Only the first 10 days and the first 5 replications of the model are shown.

            Each dot is a single patient arriving.

            From left to right within each plot, we start at one minute past midnight and move through the day until midnight.

            Looking from the top to the bottom of each plot, we have the days within a single model run.

            Each horizontal line of dots represents one **full day**.

            Hovering over the dots will show the exact time that each patient arrived and how many patients have arrived at that point in time.
            """
            )
            for i in range(5):
                st.markdown("### Model Replication {}".format(i+1))

                minimal_log = patient_log[(patient_log['event'] == 'arrival') &
                                        (patient_log['Rep'] == i+1) &
                                        (patient_log['model_day'] <=10)].sort_values("minute_in_day")

                minimal_log['model_day_str'] = minimal_log['model_day'].astype(str)


                minimal_log['minute'] = minimal_log.apply(lambda x: x['minute'] - pd.Timedelta(x['model_day'], unit="days"), axis=1)

                minimal_log['arrival_in_day'] = minimal_log.sort_values("minute").groupby('model_day')["minute"].rank()


                time_plot = px.scatter(
                        minimal_log.sort_values("minute"),
                        x="minute",
                        y="model_day_str",
                        color="model_day_str",
                        custom_data=["model_day", "minute_in_day", "arrival_in_day"],
                        category_orders={'model_day_str': [str(i+1) for i in range(5)]},
                        range_y=[0.5, min(10, n_reps)+0.5],
                        width=1200,
                        height=300,
                        opacity=0.5
                )

                del minimal_log

                time_plot.update_traces(
                    hovertemplate="<br>".join([
                        "Day: %{customdata[0]}",
                        "Time of patient arrival: %{customdata[1]}",
                        "Arrival in this day: %{customdata[2]}"
                    ])
                )

                time_plot.update_layout(yaxis_title="Model Day",
                                        xaxis_title="Time",
                            yaxis = dict(
                            tickmode = 'linear',
                            tick0 = 1,
                            dtick = 1
                        ))

                time_plot.layout.update(showlegend=False,
                                        margin=dict(l=0, r=0, t=0, b=0))

                st.plotly_chart(time_plot, use_container_width=True)

            del time_plot
            gc.collect()

        with tab3a:
            st.markdown(
                """
                The plot below shows the cumulative number of patients arriving over time for the first 5 days of each simulation run.

                Each line represents one model replication.

                Moving from left to right, we have the first 5 days of the model runs.

                Clicking on a model replication in the legend on the right-hand side of the graph will hide that line.
                Clicking on it again will make it reappear.

                By comparing the height of the two lines, you can see how similar or different the total number of patients generated are at any given point in time.


                """
            )

            minimal_log = patient_log[(patient_log['event'] == 'arrival') &
                                        (patient_log['Rep'] <=10) &
                                        (patient_log['model_day'] <=5)].sort_values("minute")
            minimal_log['cumulative_count'] = minimal_log.groupby('Rep').cumcount()

            minimal_log['Rep_str'] = minimal_log['Rep'].astype(str)

            cumulative_arrivals_fig = px.line(
                minimal_log,
                x="minute",
                y="cumulative_count",
                color="Rep_str",
                category_orders={'Rep_str': [str(i+1) for i in range(minimal_log['Rep'].max())]},
                height=800)

            hovertemplate = '%{x}: %{y} arrivals'

            # customdata = list(minimal_log[["Rep_str"]].to_numpy())
            cumulative_arrivals_fig.update_traces(
                # customdata=customdata,
                hovertemplate=hovertemplate

                )
            # del customdata
            gc.collect()

            cumulative_arrivals_fig.update_layout(xaxis_title="Model Day",
                                        yaxis_title="Cumulative Arrivals",
                                        legend_title_text='Model Replication',
                                        hovermode="x unified")

            st.plotly_chart(cumulative_arrivals_fig,
                            use_container_width=True)

            del cumulative_arrivals_fig
    gc.collect()
