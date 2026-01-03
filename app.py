import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---------------- LOAD SAVED MODELS & METADATA ---------------- #

lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
example_row = joblib.load("example_row.pkl")
metrics = joblib.load("model_metrics.pkl")

try:
    xgb_model = joblib.load("xgb_model.pkl")
    xgb_available = True
except:
    xgb_available = False

# Determine best model based on MAE
best_model_name = min(metrics, key=lambda m: metrics[m]["MAE"])


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Delhi Load Forecasting", page_icon="⚡", layout="centered")
st.title("⚡ Delhi Electricity Demand Forecasting using Machine Learning")

st.markdown("### 🏆 Best Model Based on MAE: **" + best_model_name + "**")
st.dataframe(pd.DataFrame(metrics).T.style.format("{:.2f}"))


# ---------------- WEATHER INPUT / LIVE API ---------------- #

st.markdown("---")
st.subheader("🌦 Weather Input Method")

use_live_weather = st.checkbox("Use Live Weather from API", value=False)

# initialize persistent session state variables
for key in ["temp", "rhum", "wspd", "pres"]:
    if key not in st.session_state:
        st.session_state[key] = None

if use_live_weather:
    api_key = st.text_input("Enter OpenWeatherMap API Key", type="password")
    city = st.text_input("Enter City", value="Delhi")

    if st.button("Fetch Live Weather"):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            r = requests.get(url).json()

            st.session_state.temp = r["main"]["temp"]
            st.session_state.rhum = r["main"]["humidity"]
            st.session_state.wspd = r["wind"]["speed"]
            st.session_state.pres = r["main"]["pressure"]

            st.success(f"🌤 Weather fetched successfully for {city}")

        except:
            st.error("⚠ Error fetching weather. Check API key or city name.")

    # display fetched values
    if st.session_state.temp is not None:
        st.write(f"**Temperature:** {st.session_state.temp} °C")
        st.write(f"**Humidity:** {st.session_state.rhum} %")
        st.write(f"**Wind Speed:** {st.session_state.wspd} m/s")
        st.write(f"**Pressure:** {st.session_state.pres} hPa")

else:
    st.session_state.temp = st.number_input("Temperature (°C)", step=0.5)
    st.session_state.rhum = st.number_input("Humidity (%)", step=1.0)
    st.session_state.wspd = st.number_input("Wind Speed (m/s)", step=0.1)
    st.session_state.pres = st.number_input("Pressure (hPa)", step=0.5)

# Assign to usable local variables
temp = st.session_state.temp
rhum = st.session_state.rhum
wspd = st.session_state.wspd
pres = st.session_state.pres


# ---------------- DEMAND HISTORY INPUT ---------------- #

st.markdown("---")
st.subheader("⚡ Demand Input")

lag_1 = st.number_input("Last Hour Demand (lag_1)", value=float(example_row["lag_1"]), step=10.0)
lag_2 = st.number_input("Demand 2 Hours Ago (lag_2)", value=float(example_row["lag_2"]), step=10.0)
lag_3 = st.number_input("Demand 3 Hours Ago (lag_3)", value=float(example_row["lag_3"]), step=10.0)

moving_avg_3 = (lag_1 + lag_2 + lag_3) / 3.0


# ---------------- BUILD INPUT VECTOR ---------------- #

row = example_row.copy()
row["lag_1"] = lag_1
row["lag_2"] = lag_2
row["lag_3"] = lag_3
row["moving_avg_3"] = moving_avg_3

row["temp"] = temp
row["rhum"] = rhum
row["wspd"] = wspd
row["pres"] = pres

X_input = pd.DataFrame([row[feature_cols].values], columns=feature_cols)


# ---------------- PREDICTION BUTTON ---------------- #

if st.button("🔮 Predict Next Hour Demand"):

    # Prevent model input if missing weather
    if None in [temp, rhum, wspd, pres]:
        st.error("⚠ Please fetch live weather or enter values manually before predicting.")
    else:
        # Handle NaN values
        if X_input.isnull().any().any():
            X_input = X_input.fillna(0)

        predictions = {
            "Baseline (Persistence)": lag_1,
            "Linear Regression": float(lr_model.predict(X_input)[0]),
            "Random Forest": float(rf_model.predict(X_input)[0])
        }

        if xgb_available:
            predictions["XGBoost"] = float(xgb_model.predict(X_input)[0])

        st.subheader("📊 Model Predictions (kW)")
        pred_df = pd.DataFrame(predictions.items(), columns=["Model", "kW"])
        pred_df["Best"] = pred_df["Model"].apply(lambda m: "⭐ Best" if m == best_model_name else "")

        st.dataframe(pred_df.style.format({"kW": "{:.2f}"}))

        # Bar chart
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(pred_df["Model"], pred_df["kW"])
        ax.set_ylabel("Power Demand (kW)")
        ax.set_title("Model Prediction Comparison")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        st.success(f"🏆 Best Model for this Dataset: **{best_model_name}**")
