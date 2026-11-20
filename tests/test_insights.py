"""Run with:  python -m pytest tests -q"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))

import insights as I  # noqa: E402

FLAGS = [
    "360_camera", "abs", "air_bags", "air_conditioning", "alloy_wheels",
    "android_auto", "apple_carplay", "climate_control", "cruise_control",
    "drls", "front_camera", "front_fog_lights", "front_speakers",
    "head_up_display_(hud)", "heated_seats", "immobilizer_key",
    "infotainment_system", "keyless_entry", "led_headlights",
    "paddle_shifters", "panoramic_sunroof", "parking_sensors", "power_locks",
    "power_mirrors", "power_seats", "power_steering", "push_start",
    "rear_ac_vents", "rear_camera", "rear_speakers", "steering_switches",
    "sun_roof", "traction_control", "tyre_pressure_monitoring_system_(tpms)",
    "ventilated_seats",
]


@pytest.fixture(scope="module")
def model():
    return joblib.load(os.path.join(ROOT, "models", "pak_car_price_xgboost.pkl"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(os.path.join(ROOT, "data", "processed", "cleaned_cars.csv"))


def make_car(**overrides):
    car = dict(
        registered_city="Lahore", color="White", assembly="Local",
        body_type="Sedan", brand="Toyota", model="Corolla",
        variant="Altis Grande CVT-i 1.8", engine_cc=1800, year=2021,
        **{f: 0 for f in FLAGS},
    )
    car.update(overrides)
    return car


def test_explanation_adds_up_to_prediction(model):
    X = pd.DataFrame([make_car(abs=1, cruise_control=1)])
    price = model.predict(X)[0]
    baseline, effects = I.explain(model, X)
    total = np.log1p(baseline) + effects["log_contrib"].sum()
    assert np.isclose(total, np.log1p(price), atol=1e-3)


def test_newer_car_is_worth_more(model):
    old = model.predict(pd.DataFrame([make_car(year=2015)]))[0]
    new = model.predict(pd.DataFrame([make_car(year=2023)]))[0]
    assert new > old


def test_every_variant_in_data_predicts(model, data):
    for (brand, mdl, variant), group in data.groupby(["brand", "model", "variant"]):
        cc, body = I.default_specs(data, brand, mdl, variant)
        car = make_car(brand=brand, model=mdl, variant=variant, engine_cc=cc,
                       body_type=body or "Sedan", year=int(group["year"].median()))
        price = model.predict(pd.DataFrame([car]))[0]
        assert np.isfinite(price) and price > 0, (brand, mdl, variant)


def test_unknown_car_does_not_crash(model, data):
    car = make_car(brand="Tesla", model="Model S", variant="Plaid")
    assert model.predict(pd.DataFrame([car]))[0] > 0
    assert I.similar_cars(data, car).empty


def test_similar_cars_same_model_nearest_year(data):
    car = make_car()
    similar = I.similar_cars(data, car)
    assert 0 < len(similar) <= 5
    assert (similar["Year"] - 2021).abs().max() <= 3


def test_default_specs_for_corolla_grande(data):
    cc, body = I.default_specs(data, "Toyota", "Corolla", "Altis Grande CVT-i 1.8")
    assert cc == 1800 and body == "Sedan"


def test_original_column_mapping():
    assert I._original_column("cat__body_type_Sedan") == "body_type"
    assert I._original_column("cat__model_Corolla") == "model"
    assert I._original_column("num__abs") == "abs"
