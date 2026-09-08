# Streamlit app for the Crazy Lift Taxi demand forecast

import streamlit as st
import pandas as pd
from src.data import load_data, resample_hourly
from src.predict import load_model, predict_next_hour
import config

st.set_page_config(page_title="Crazy Lift Taxi Demand Forecast", page_icon="🚕")

st.title("🚕 Crazy Lift Taxi Demand Forecast")
st.write("Predict how many taxi orders to expect at the airport in the next hour.")

# Load the trained model and its expected feature order
model, feature_columns = load_model()

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

@st.cache_data
def get_recent_history():
    # Use the last real hourly order counts in the dataset as a sensible default
    data = resample_hourly(load_data())
    return data[config.TARGET_COL].tail(config.MAX_LAG).tolist(), data.index[-1]

default_history, last_timestamp = get_recent_history()

st.header("Last 24 hours of order counts")
st.caption("Oldest hour first, most recent hour last. Defaults are the last real values in the dataset, edit them to try other scenarios.")

history_df = pd.DataFrame({'orders': default_history})
edited_df = st.data_editor(history_df, hide_index=True, num_rows="fixed")
history = edited_df['orders'].tolist()

st.header("Hour to predict")
col1, col2 = st.columns(2)

with col1:
    dayofweek = st.selectbox(
        "Day of week",
        options=list(range(7)),
        format_func=lambda d: DAY_NAMES[d],
        index=int(last_timestamp.dayofweek)
    )

with col2:
    hour = st.selectbox(
        "Hour of day",
        options=list(range(24)),
        format_func=lambda h: f"{h:02d}:00",
        index=int((last_timestamp.hour + 1) % 24)
    )

if st.button("Predict next hour"):
    prediction = predict_next_hour(model, feature_columns, history, dayofweek, hour)
    st.success(f"Estimated taxi orders in the next hour: {prediction}")
