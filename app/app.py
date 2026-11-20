import html
import os
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insights import (  # noqa: E402
    default_specs,
    explain,
    load_metrics,
    pretty_feature,
    similar_cars,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(ROOT, "models", "pak_car_price_xgboost.pkl")
DATA_PATH = os.path.join(ROOT, "data", "processed", "cleaned_cars.csv")
GITHUB_URL = "https://github.com/hassamjaved83-cell/pakcar-price-predictor"

FEATURE_COLUMNS = [
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

st.set_page_config(
    page_title="PakCar Price Predictor",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


model = load_model(MODEL_PATH)
df = load_data(DATA_PATH)
metrics = load_metrics(ROOT)

R2 = metrics.get("test_r2", 0.974)
MAPE = metrics.get("test_mape_pct", 8.0)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero h1 {font-size: 2.4rem; line-height: 1.15; margin: 0 0 .35rem 0; letter-spacing: -.02em;}
    .hero p {color: #9ca3af; font-size: 1.02rem; margin: 0 0 1rem 0;}
    .chips {display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.4rem;}
    .chip {border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04);
           border-radius: 999px; padding: .35rem .85rem; font-size: .88rem; color: #e5e7eb;}
    .chip b {color: #34d399;}
    .result {border-radius: 18px; padding: 1.4rem 1.5rem;
             background: linear-gradient(135deg, #064e3b 0%, #065f46 45%, #0f766e 100%);
             border: 1px solid rgba(52,211,153,.35); margin-bottom: 1rem;}
    .result .label {color: #a7f3d0; font-size: .85rem; text-transform: uppercase; letter-spacing: .08em;}
    .result .lakh {font-size: 3rem; font-weight: 800; line-height: 1.1; color: #fff; margin: .2rem 0;}
    .result .pkr {color: #d1fae5; font-size: 1.05rem;}
    .result .range {margin-top: .7rem; color: #ecfdf5; font-size: .95rem;}
    .placeholder {border: 1px dashed rgba(255,255,255,.18); border-radius: 18px; padding: 2.2rem 1.5rem;
                  text-align: center; color: #9ca3af;}
    .placeholder .big {font-size: 2.2rem; margin-bottom: .4rem;}
    table.sim {width: 100%; border-collapse: collapse; font-size: .9rem; margin-bottom: .8rem;}
    table.sim th {text-align: left; color: #9ca3af; font-weight: 600; padding: .45rem .5rem;
                  border-bottom: 1px solid rgba(255,255,255,.12);}
    table.sim td {padding: .45rem .5rem; border-bottom: 1px solid rgba(255,255,255,.06);
                  color: #e5e7eb; white-space: nowrap;}
    table.sim td:first-child {white-space: normal; overflow-wrap: break-word;}
    table.sim .p {text-align: right; font-weight: 700; color: #34d399; white-space: nowrap;}
    .footer {color: #6b7280; font-size: .85rem; text-align: center; margin-top: 1.5rem;}
    .footer a {color: #34d399; text-decoration: none;}
    @media (max-width: 640px) {
        .hero h1 {font-size: 1.8rem;}
        .result .lakh {font-size: 2.3rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    f"""
    <div class="hero">
      <h1>🚗 PakCar Price Predictor</h1>
      <p>What is a used car really worth in Pakistan? Pick a car and the
      XGBoost model estimates its market price, explains why, and shows
      real listings to compare.</p>
    </div>
    <div class="chips">
      <span class="chip"><b>7,835</b> PakWheels ads scraped</span>
      <span class="chip">R² <b>{R2:.3f}</b> on unseen cars</span>
      <span class="chip"><b>±{MAPE:.0f}%</b> average error</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def options(series):
    return sorted(series.dropna().astype(str).unique())


left, right = st.columns([5, 4], gap="large")


# =========================================================
# INPUTS
# =========================================================

with left:
    with st.container(border=True):
        st.markdown("#### Car details")

        c1, c2, c3 = st.columns(3)
        with c1:
            brands = options(df["brand"])
            brand = st.selectbox("Brand", brands, index=brands.index("Toyota") if "Toyota" in brands else 0)
        with c2:
            models = options(df[df["brand"] == brand]["model"])
            model_name = st.selectbox("Model", models)
        with c3:
            variants = options(df[(df["brand"] == brand) & (df["model"] == model_name)]["variant"]) or ["Unknown"]
            variant = st.selectbox("Variant", variants)

        default_cc, default_body = default_specs(df, brand, model_name, variant)
        body_options = options(df["body_type"])

        c4, c5, c6 = st.columns(3)
        with c4:
            year = st.number_input(
                "Year",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=2021,
                step=1,
            )
        with c5:
            engine_cc = st.number_input(
                "Engine (cc)",
                min_value=600,
                max_value=6000,
                value=max(600, min(6000, default_cc)),
                step=100,
                key=f"cc::{brand}::{model_name}::{variant}",
            )
        with c6:
            body_type = st.selectbox(
                "Body type",
                body_options,
                index=body_options.index(default_body) if default_body in body_options else 0,
                key=f"body::{brand}::{model_name}::{variant}",
            )

        cities = options(df["registered_city"])
        colors = options(df["color"])
        assemblies = options(df["assembly"])

        c7, c8, c9 = st.columns(3)
        with c7:
            city = st.selectbox("Registered city", cities, index=cities.index("Lahore") if "Lahore" in cities else 0)
        with c8:
            color = st.selectbox("Colour", colors, index=colors.index("White") if "White" in colors else 0)
        with c9:
            assembly = st.selectbox("Assembly", assemblies, index=assemblies.index("Local") if "Local" in assemblies else 0)

        feature_labels = {pretty_feature(f): f for f in FEATURE_COLUMNS}
        chosen = st.multiselect(
            "Features in the car",
            list(feature_labels),
            placeholder="ABS, airbags, sunroof, cruise control…",
        )

        predict_clicked = st.button("Estimate price", type="primary", width="stretch")


car_data = {
    "registered_city": city,
    "color": color,
    "assembly": assembly,
    "body_type": body_type,
    "brand": brand,
    "model": model_name,
    "variant": variant,
    "engine_cc": int(engine_cc),
    "year": int(year),
    **{f: int(f in {feature_labels[c] for c in chosen}) for f in FEATURE_COLUMNS},
}
car_key = tuple(sorted(car_data.items()))

if predict_clicked:
    st.session_state["result_for"] = car_key

show_result = st.session_state.get("result_for") == car_key


# =========================================================
# RESULT
# =========================================================

with right:
    if not show_result:
        st.markdown(
            """
            <div class="placeholder">
              <div class="big">🔍</div>
              Choose a car on the left and press <b>Estimate price</b>.<br>
              You'll get the price, the reasons behind it, and real listings to compare.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            input_df = pd.DataFrame([car_data])
            price = float(model.predict(input_df)[0])
            low, high = price * (1 - MAPE / 100), price * (1 + MAPE / 100)

            st.markdown(
                f"""
                <div class="result">
                  <div class="label">Estimated market price</div>
                  <div class="lakh">{price / 100_000:,.1f} lakh</div>
                  <div class="pkr">PKR {price:,.0f}</div>
                  <div class="range">Likely range: <b>{low / 100_000:,.1f} – {high / 100_000:,.1f} lakh</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- Why this price ----------
            _, effects = explain(model, input_df)
            # Show what the user actually chose; fold "missing feature" effects into one line
            shown = effects[~effects["label"].str.startswith("No ")].head(6).copy()
            missing = effects[effects["label"].str.startswith("No ")]["log_contrib"].sum()
            if abs(missing) > 0.01:
                shown = pd.concat(
                    [shown, pd.DataFrame([{
                        "label": "Features not selected",
                        "effect_pct": (np.exp(missing) - 1) * 100,
                    }])],
                    ignore_index=True,
                )
            shown["text"] = shown["effect_pct"].map(lambda v: f"{v:+.0f}%")
            shown["direction"] = shown["effect_pct"].map(lambda v: "raises" if v >= 0 else "lowers")

            st.markdown("#### Why this price?")
            st.caption("How much each input moves the price compared with an average car in the data.")

            # Leave room on both sides so the % labels never get cut off
            lo = min(0.0, shown["effect_pct"].min())
            hi = max(0.0, shown["effect_pct"].max())
            pad = max(4.0, (hi - lo) * 0.22)
            domain = [lo - (pad if lo < 0 else 0), hi + pad]

            base = alt.Chart(shown).encode(
                y=alt.Y(
                    "label:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=190, labelFontSize=12, labelColor="#d1d5db", ticks=False, domain=False),
                ),
            )
            bars = base.mark_bar(cornerRadius=4, height=18).encode(
                x=alt.X(
                    "effect_pct:Q", title=None, scale=alt.Scale(domain=domain),
                    axis=alt.Axis(format="+.0f", labelExpr="datum.label + '%'", labelColor="#9ca3af", gridColor="#1f2937"),
                ),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["raises", "lowers"], range=["#34d399", "#f87171"]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("text:N", title="Effect")],
            )
            text_up = base.transform_filter(alt.datum.effect_pct >= 0).mark_text(
                align="left", dx=5, fontSize=12, fontWeight="bold", color="#e5e7eb"
            ).encode(x="effect_pct:Q", text="text:N")
            text_down = base.transform_filter(alt.datum.effect_pct < 0).mark_text(
                align="right", dx=-5, fontSize=12, fontWeight="bold", color="#e5e7eb"
            ).encode(x="effect_pct:Q", text="text:N")

            st.altair_chart(
                (bars + text_up + text_down)
                .properties(height=34 * len(shown) + 10)
                .configure_view(stroke=None),
                width="stretch",
            )

            # ---------- Similar cars ----------
            similar = similar_cars(df, car_data)
            if not similar.empty:
                st.markdown("#### Similar cars on PakWheels")
                rows = "".join(
                    f"<tr><td>{html.escape(str(r['Variant']))}</td><td>{r['Year']}</td>"
                    f"<td>{html.escape(str(r['City']))}</td><td class='p'>{r['Price (lakh)']:.1f}</td></tr>"
                    for _, r in similar.iterrows()
                )
                st.markdown(
                    f"""
                    <table class="sim">
                      <thead><tr><th>Variant</th><th>Year</th><th>City</th><th class="p">Lakh</th></tr></thead>
                      <tbody>{rows}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Estimate from listing data. Mileage, condition, accident history "
                "and paperwork can move the real price."
            )

        except Exception as e:  # keep the page usable if a rare input breaks
            st.error("Couldn't estimate this car. Try another variant or year.")
            st.exception(e)


st.markdown(
    f"""
    <div class="footer">
      Built by Hassam Javed · Python, Playwright, XGBoost, Streamlit ·
      <a href="{GITHUB_URL}" target="_blank">Code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
