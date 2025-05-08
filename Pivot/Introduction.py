import gc
import streamlit as st
from helper_functions import add_logo, mermaid

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)

add_logo()

with open("style.css") as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.title("Welcome to the Discrete Event Simulation Playground! 👋")

gc.collect()

st.markdown(
"""
This is a discrete event simulation playground based on the Monks et al (2022), which is itself an implementation of the Treatment Centre Model from Nelson (2013).

Project Overview: Hospital Simulation with AI-Driven Resource Management

    - Purpose: Model and optimize patient flow in a hospital using simulation and reinforcement learning.

    - Method: Combines discrete event simulation (SimPy) with PPO/RAPPO agents for decision-making.

    - Focus: AI controls staffing and resource allocation per shift, adapting to unpredictable patient demand.

    - Objective: Reduce wait times, avoid overcrowding, and improve overall efficiency under real-world constraints.

    - Tools: SimPy (for simulation), pyTorch (for RL), Streamlit + Plotly (for visualization).

    - Outcome: A dynamic testbed for experimenting with smarter, data-driven hospital operations.


"""
)


mermaid(height=800
        
        , code=
"""
  %%{ init: {
        "flowchart": { "curve": "step" },
        "theme": "base",
        "themeVariables": { "lineColor": "#b4b4b4" }
        }}%%
        flowchart LR

        A[Arrival] --> BX[Triage]
        BX -.-> T([Triage Bay\nRESOURCE])
        T -.-> BX

        BX --> B{Trauma or Non-Trauma}

        %% TRAUMA BRANCH
        B --> TP[Trauma Pathway]
        TP --> C[Stabilisation]
        C --> E[Trauma Treatment]
        C -.-> Z([Trauma Room\nRESOURCE])
        Z -.-> C
        E -.-> Y([Cubicle 2\nRESOURCE])
        Y -.-> E

        %% Trauma outcome split
        E --> DT[Discharge]
        E --> WT[Ward Admission]
        E --> IT[ICU Admission]
        WT --> WTB([Ward Beds])
        IT --> ICU2([ICU Beds])

        %% NON-TRAUMA BRANCH
        B --> NP[Non-Trauma Pathway]
        NP --> D[Registration]
        D -.-> X([Clerks\nRESOURCE])
        X -.-> D
        D --> G[Examination]
        G -.-> W([Exam Room\nRESOURCE])
        W -.-> G
        G --> H{Needs Treatment?}

        H --> DN[Discharge]
        H --> I[Non-Trauma Treatment]
        I -.-> V([Cubicle 1\nRESOURCE])
        V -.-> I

        %% Non-Trauma outcome split
        I --> DN2[Discharge]
        I --> WN[Ward Admission]
        I --> IN[ICU Admission]
        WN --> WNB([Ward Beds])
        IN --> ICU1([ICU Beds])

        %% Node styling
        classDef res fill:#e6f7ff,stroke:#007acc,stroke-width:2px,color:#003366;
        classDef start fill:#02CD55,color:#fff,font-weight:bold;
        classDef path fill:#ccc,stroke:#666;

        class A start;
        class Z,Y,X,W,V,WTB,ICU2,WNB,ICU1 res;
        class TP,NP,C,E,D,G,H,I path;


     
    """
)

