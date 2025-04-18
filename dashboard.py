import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Caching data and model to optimize performance
def load_data(path: str = 'X_val1.csv') -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str = 'model3.pkl'):
    """Load the trained fraud-detection model."""
    return joblib.load(path)

# Main application

def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.title("💳 Fraud Detection Dashboard")

    # Sidebar controls
    st.sidebar.header("Settings")
    data_path = st.sidebar.text_input("Test data CSV path", value="X_val1.csv")
    model_path = st.sidebar.text_input("Trained model path", value="model3.pkl")
    show_raw = st.sidebar.checkbox("Show raw data", value=False)

    # Load data & model
    df = load_data(data_path)
    model = load_model(model_path)

    # Preprocess time for trends (if 'Time' exists)
    if 'Time' in df.columns:
        df['Time_hours'] = df['Time'] / 3600
    else:
        df['Time_hours'] = df.index

    # Generate predictions/anomaly flags
    features = df.drop(columns=[ 'Time_hours'], errors='ignore')
    df['Anomaly'] = model.predict(features)

    # Show raw data if requested
    if show_raw:
        st.subheader("Raw Test Data")
        st.dataframe(df)

    # Transaction trends over time
    st.subheader("🔍 Transaction Trends Over Time")
    trend_df = df.copy()
    trend_df['Time_bin'] = pd.cut(trend_df['Time_hours'], bins=50)
    counts = trend_df.groupby('Time_bin').size().reset_index(name='Transactions')
    counts['Time_mid'] = counts['Time_bin'].apply(lambda x: x.mid)
    trend_chart = alt.Chart(counts).mark_line(point=True).encode(
        x=alt.X('Time_mid', title='Time (hours)'),
        y=alt.Y('Transactions', title='Number of Transactions')
    ).properties(width=700, height=400)
    st.altair_chart(trend_chart, use_container_width=True)

    # Flagged anomalies scatter
    st.subheader("🚩 Flagged Anomalies")
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('Time_hours', title='Time (hours)'),
        y=alt.Y('Amount', title='Transaction Amount'),
        color=alt.Color('Anomaly:N', title='Anomaly', scale=alt.Scale(domain=[0,1], range=['#1f77b4', '#d62728']))
    ).properties(width=700, height=400)
    st.altair_chart(scatter, use_container_width=True)

    # Table of anomalies
    st.subheader("Flagged Anomaly Records")
    anomalies = df[df['Anomaly'] == 1]
    st.write(f"Total anomalies flagged: {len(anomalies)}")
    st.dataframe(anomalies)

if __name__ == '__main__':
    main()
