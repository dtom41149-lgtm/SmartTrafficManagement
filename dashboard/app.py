import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "traffic_density_model.pkl"
DATA_PATH = PROJECT_DIR / "results" / "cleaned_traffic_data.csv"
METRICS_PATH = PROJECT_DIR / "results" / "model_metrics.txt"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🚦 Smart Traffic Management System")

st.write(
    "A Big Data and Machine Learning based system "
    "for predicting urban traffic density."
)

st.divider()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Features",
        "11"
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Traffic Density Mean",
        f"{df['Traffic Density'].mean():.4f}"
    )


st.write(
    "The dataset contains more than 1.2 million traffic records "
    "covering environmental, vehicle, economic and time-related factors."
)


# ============================================================
# TRAFFIC ANALYTICS
# ============================================================

st.divider()

st.header("📈 Traffic Analytics")

# Hourly analysis
hourly = (
    df.groupby("Hour Of Day")
    .agg(
        Average_Traffic_Density=("Traffic Density", "mean"),
        Average_Speed=("Speed", "mean")
    )
    .reset_index()
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Traffic by Hour",
        "Speed by Hour",
        "Peak Analysis",
        "Traffic Distribution"
    ]
)


with tab1:

    st.subheader("Average Traffic Density by Hour")

    chart_data = hourly.set_index("Hour Of Day")[
        ["Average_Traffic_Density"]
    ]

    st.line_chart(chart_data)


with tab2:

    st.subheader("Average Speed by Hour")

    chart_data = hourly.set_index("Hour Of Day")[
        ["Average_Speed"]
    ]

    st.line_chart(chart_data)


with tab3:

    st.subheader("Peak vs Non-Peak Traffic")

    peak_data = (
        df.groupby("Is Peak Hour")["Traffic Density"]
        .mean()
        .rename(index={
            0: "Non-Peak",
            1: "Peak"
        })
    )

    st.bar_chart(peak_data)


with tab4:

    st.subheader("Traffic Density Distribution")

    histogram_data = pd.DataFrame({
        "Traffic Density": df["Traffic Density"]
    })

    st.bar_chart(
        histogram_data["Traffic Density"]
        .value_counts(bins=20)
        .sort_index()
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("🤖 Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "R² Score",
        "0.759152"
    )

with metric2:
    st.metric(
        "MAE",
        "0.069262"
    )

with metric3:
    st.metric(
        "RMSE",
        "0.107630"
    )


st.write(
    "**Model:** HistGradientBoostingRegressor"
)

st.info(
    "R² indicates how much variation in Traffic Density "
    "is explained by the model. MAE and RMSE measure prediction error."
)


# ============================================================
# TRAFFIC PREDICTION
# ============================================================

st.divider()

st.header("🚦 Predict Traffic Density")

st.write(
    "Enter the traffic conditions below to generate a prediction."
)


col1, col2, col3 = st.columns(3)


with col1:

    city = st.selectbox(
        "City",
        [
            "Ecopolis",
            "AquaCity",
            "Neuroburg",
            "SolarisVille",
            "MetropolisX",
            "TechHaven"
        ]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        [
            "Autonomous Vehicle",
            "Drone",
            "Flying Car",
            "Car"
        ]
    )

    weather = st.selectbox(
        "Weather",
        [
            "Solar Flare",
            "Snowy",
            "Electromagnetic Storm",
            "Clear",
            "Rainy"
        ]
    )


with col2:

    economic_condition = st.selectbox(
        "Economic Condition",
        [
            "Booming",
            "Recession",
            "Stable"
        ]
    )

    day_of_week = st.selectbox(
        "Day Of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    hour = st.slider(
        "Hour Of Day",
        min_value=0,
        max_value=23,
        value=12
    )


with col3:

    speed = st.number_input(
        "Speed",
        min_value=0.0,
        max_value=200.0,
        value=60.0,
        step=1.0
    )

    is_peak_hour = st.selectbox(
        "Is Peak Hour?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    random_event = st.selectbox(
        "Random Event Occurred?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# FEATURE ENGINEERING FOR USER INPUT
# ============================================================

hour_sin = np.sin(2 * np.pi * hour / 24)

hour_cos = np.cos(2 * np.pi * hour / 24)


if 6 <= hour < 10:

    time_period = "Morning Peak"

elif 10 <= hour < 16:

    time_period = "Daytime"

elif 16 <= hour < 20:

    time_period = "Evening Peak"

elif 20 <= hour < 24:

    time_period = "Night"

else:

    time_period = "Late Night"


speed_peak_interaction = speed * is_peak_hour


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🚦 Predict Traffic Density",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "City": [city],

        "Vehicle Type": [vehicle_type],

        "Weather": [weather],

        "Economic Condition": [economic_condition],

        "Day Of Week": [day_of_week],

        "Time Period": [time_period],

        "Hour Of Day": [hour],

        "Hour_Sin": [hour_sin],

        "Hour_Cos": [hour_cos],

        "Speed": [speed],

        "Is Peak Hour": [is_peak_hour],

        "Random Event Occurred": [random_event],

        "Speed_Peak_Interaction": [
            speed_peak_interaction
        ]
    })


    prediction = float(
        model.predict(input_data)[0]
    )


    st.subheader("Prediction Result")


    result1, result2 = st.columns(2)


    with result1:

        st.metric(
            "Predicted Traffic Density",
            f"{prediction:.4f}"
        )


    with result2:

        if prediction < 0.15:

            level = "Low Traffic"

        elif prediction < 0.30:

            level = "Moderate Traffic"

        else:

            level = "High Traffic"


        st.metric(
            "Traffic Level",
            level
        )


    if prediction < 0.15:

        st.success(
            "Traffic density is predicted to be low."
        )

    elif prediction < 0.30:

        st.warning(
            "Traffic density is predicted to be moderate."
        )

    else:

        st.error(
            "Traffic density is predicted to be high."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Traffic Management System | "
    "PySpark + Machine Learning + Streamlit"
)