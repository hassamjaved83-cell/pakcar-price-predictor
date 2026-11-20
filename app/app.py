import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PakCar Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "pak_car_price_xgboost.pkl"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(MODEL_PATH)


# =========================================================
# LOAD DATA FOR DROPDOWN OPTIONS
# =========================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "cleaned_cars.csv"
)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data(DATA_PATH)


# =========================================================
# HEADER
# =========================================================

st.title("🚗 PakCar Price Predictor")

st.markdown(
    """
    ### 🇵🇰 Pakistani Used Car Price Prediction

    Enter the details of a car below and our trained
    **XGBoost Machine Learning model** will estimate its
    market price in Pakistani Rupees.
    """
)

st.divider()


# =========================================================
# BASIC CAR INFORMATION
# =========================================================

st.subheader("🚘 Car Information")

col1, col2, col3 = st.columns(3)


with col1:

    brand = st.selectbox(
        "Brand",
        sorted(df["brand"].dropna().unique())
    )

    model_name = st.selectbox(
        "Model",
        sorted(
            df[
                df["brand"] == brand
            ]["model"]
            .dropna()
            .unique()
        )
    )

    year = st.number_input(
        "Year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=2021,
        step=1
    )


with col2:

    variant_options = sorted(
        df[
            (df["brand"] == brand)
            &
            (df["model"] == model_name)
        ]["variant"]
        .dropna()
        .unique()
    )

    variant = st.selectbox(
        "Variant",
        variant_options
        if len(variant_options) > 0
        else ["Unknown"]
    )

    engine_cc = st.number_input(
        "Engine Capacity (CC)",
        min_value=500,
        max_value=10000,
        value=1300,
        step=100
    )

    body_type_options = sorted(
        df["body_type"]
        .dropna()
        .unique()
    )

    body_type = st.selectbox(
        "Body Type",
        body_type_options
    )


with col3:

    city = st.selectbox(
        "Registered City",
        sorted(
            df["registered_city"]
            .dropna()
            .unique()
        )
    )

    color = st.selectbox(
        "Color",
        sorted(
            df["color"]
            .dropna()
            .unique()
        )
    )

    assembly = st.selectbox(
        "Assembly",
        sorted(
            df["assembly"]
            .dropna()
            .unique()
        )
    )


# =========================================================
# FEATURES
# =========================================================

st.divider()

st.subheader("⚙️ Car Features")

st.caption(
    "Select the features available in the vehicle."
)


feature_columns = [

    "360_camera",
    "abs",
    "air_bags",
    "air_conditioning",
    "alloy_wheels",
    "android_auto",
    "apple_carplay",
    "climate_control",
    "cruise_control",
    "drls",
    "front_camera",
    "front_fog_lights",
    "front_speakers",
    "head_up_display_(hud)",
    "heated_seats",
    "immobilizer_key",
    "infotainment_system",
    "keyless_entry",
    "led_headlights",
    "paddle_shifters",
    "panoramic_sunroof",
    "parking_sensors",
    "power_locks",
    "power_mirrors",
    "power_seats",
    "power_steering",
    "push_start",
    "rear_ac_vents",
    "rear_camera",
    "rear_speakers",
    "steering_switches",
    "sun_roof",
    "traction_control",
    "tyre_pressure_monitoring_system_(tpms)",
    "ventilated_seats"

]


feature_values = {}


# Create 3 columns for feature checkboxes

feature_cols = st.columns(3)


for index, feature in enumerate(feature_columns):

    display_name = (
        feature
        .replace("_", " ")
        .replace("(tpms)", "TPMS")
        .title()
    )

    with feature_cols[index % 3]:

        feature_values[feature] = int(
            st.checkbox(
                display_name,
                value=False
            )
        )


# =========================================================
# BUILD INPUT
# =========================================================

car_data = {

    "registered_city": city,
    "color": color,
    "assembly": assembly,
    "body_type": body_type,

    "brand": brand,
    "model": model_name,
    "variant": variant,

    "engine_cc": engine_cc,
    "year": year,

    **feature_values

}


# =========================================================
# PREDICTION
# =========================================================

st.divider()

st.subheader("💰 Get Estimated Price")

if st.button(
    "🚀 Predict Car Price",
    use_container_width=True
):

    try:

        # Build dataframe
        input_df = pd.DataFrame([car_data])

        # Make prediction
        prediction = model.predict(input_df)

        prediction = float(prediction[0])

        # Display result
        st.success("Prediction generated successfully!")

        st.markdown(
            f"""
            ## 💰 Estimated Market Price

            # PKR {prediction:,.0f}

            ### ≈ {prediction / 100_000:,.1f} lakh
            """
        )

        st.info(
            "This is an ML-based estimated market price. "
            "Actual prices may vary depending on mileage, "
            "condition, accident history, documentation, "
            "location and other factors."
        )

    except Exception as e:

        st.error("❌ Prediction failed")

        st.exception(e)
# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PakCar Price Predictor • Powered by XGBoost Machine Learning"
)