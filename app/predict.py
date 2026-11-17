import os

import joblib
import pandas as pd


# Load trained model (path works from any working directory)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "models",
    "pak_car_price_xgboost.pkl",
)

model = joblib.load(MODEL_PATH)


def predict_price(car_data):
    """
    Predict the price of one car.

    car_data should be a dictionary containing
    the same input features used during training.
    """

    car_df = pd.DataFrame([car_data])

    prediction = model.predict(car_df)[0]

    return round(float(prediction), 2)


if __name__ == "__main__":

    sample_car = {
        "registered_city": "Lahore",
        "color": "White",
        "assembly": "Local",
        "body_type": "Sedan",
        "brand": "Toyota",
        "model": "Corolla",
        "variant": "Grande",
        "engine_cc": 1800,
        "year": 2021,

        "360_camera": 0,
        "abs": 1,
        "air_bags": 1,
        "air_conditioning": 1,
        "alloy_wheels": 1,
        "android_auto": 1,
        "apple_carplay": 1,
        "climate_control": 1,
        "cruise_control": 1,
        "drls": 1,
        "front_camera": 1,
        "front_fog_lights": 1,
        "front_speakers": 1,
        "head_up_display_(hud)": 0,
        "heated_seats": 0,
        "immobilizer_key": 1,
        "infotainment_system": 1,
        "keyless_entry": 1,
        "led_headlights": 1,
        "paddle_shifters": 0,
        "panoramic_sunroof": 0,
        "parking_sensors": 1,
        "power_locks": 1,
        "power_mirrors": 1,
        "power_seats": 1,
        "power_steering": 1,
        "push_start": 1,
        "rear_ac_vents": 1,
        "rear_camera": 1,
        "rear_speakers": 1,
        "steering_switches": 1,
        "sun_roof": 0,
        "traction_control": 1,
        "tyre_pressure_monitoring_system_(tpms)": 1,
        "ventilated_seats": 0
    }

    price = predict_price(sample_car)

    print(f"Predicted Price: PKR {price:,.0f}")