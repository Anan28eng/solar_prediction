import streamlit as st
import matplotlib.pyplot as plt
from pipeline import load_data, clean_data, add_features, get_features_and_target, split_data, train_xgb, evaluate_model, save_model, load_model
import os

MODEL_PATH = "models/xgb_model.joblib"
CSV_PATH = "solar_data_cleaned.csv"

st.set_page_config(page_title="Solar Production Dashboard", layout="wide")

st.title("Solar System Production — Model Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Train / Retrain Model"):
    with st.spinner("Training model (this may take a while)..."):
        try:
            result = None
            # Run the pipeline build and train
            from pipeline import build_and_train
            result = build_and_train(csv_path=CSV_PATH, save_path=MODEL_PATH)
            st.success("Model trained and saved.")
            metrics = result["metrics"]
            st.sidebar.metric("RMSE", f"{metrics['rmse']:.4f}")
            st.sidebar.metric("R2", f"{metrics['r2']:.4f}")
        except Exception as e:
            st.error(f"Training failed: {e}")

st.sidebar.markdown("---")

st.sidebar.header("Model Info")
if os.path.exists(MODEL_PATH):
    st.sidebar.write(f"Model file: `{MODEL_PATH}`")
else:
    st.sidebar.write("No trained model found. Click 'Train / Retrain Model' to create one.")

# Load data and show sample
st.header("Data Preview")
try:
    df = load_data(CSV_PATH)
    st.dataframe(df.head(100))
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# Prepare features and split
df = clean_data(df)
df = add_features(df)
try:
    X, y = get_features_and_target(df)
except Exception as e:
    st.error(f"Feature extraction failed: {e}")
    st.stop()

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Prediction & evaluation
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    st.header("Model Performance on Holdout Set")
    cols = st.columns(4)
    cols[0].metric("RMSE", f"{metrics['rmse']:.4f}")
    cols[1].metric("MAE", f"{metrics['mae']:.4f}")
    cols[2].metric("MSE", f"{metrics['mse']:.4f}")
    cols[3].metric("R2", f"{metrics['r2']:.4f}")

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.index, y_test.values, label="Actual", alpha=0.8)
    ax.plot(y_test.index, y_pred, label="Predicted", alpha=0.8)
    ax.legend()
    st.pyplot(fig)

    # Suggested Appliance Run Windows based on predicted power threshold
    st.subheader("Suggested Appliance Run Windows")
    try:
        threshold = float(y_pred.mean())
        high_power_times = y_test.index[y_pred > threshold]
        st.write("Threshold (mean predicted power):", f"{threshold:.4f}")
        if len(high_power_times) == 0:
            st.write("No high-power windows found based on the threshold.")
        else:
            # show a sample of times and allow user to expand
            st.write("Number of high-power timepoints:", len(high_power_times))
            st.write(high_power_times.to_series().head(50))
    except Exception as e:
        st.info(f"Could not compute appliance run windows: {e}")

    # Battery Behavior Simulation (simple cumulative logic)
    st.subheader("Battery Behavior Simulation")
    try:
        # allow user to tune simple params in sidebar
        battery_capacity = st.sidebar.number_input("Battery capacity (kWh)", value=10.0, min_value=0.0, step=1.0)
        initial_charge = st.sidebar.number_input("Initial battery charge (kWh)", value=5.0, min_value=0.0, max_value=battery_capacity, step=1.0)
        discharge_threshold = st.sidebar.number_input("Discharge threshold (kW)", value=2.0, min_value=0.0, step=0.5)

        battery_levels = []
        battery_charge = float(initial_charge)
        for power in y_pred:
            # if predicted power above discharge_threshold, charge by surplus, else no charge
            battery_charge = min(battery_capacity, battery_charge + max(0.0, float(power) - float(discharge_threshold)))
            battery_levels.append(battery_charge)

        # plot battery levels
        fig3, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(y_test.index, battery_levels, label='Battery Level')
        ax3.set_xlabel('Date-Time')
        ax3.set_ylabel('Battery Charge (kWh)')
        ax3.set_title('Battery Behavior Simulation')
        ax3.legend()
        st.pyplot(fig3)

        # expose some summary stats
        import numpy as _np
        st.write('Final battery charge (kWh):', f"{_np.round(battery_levels[-1], 3)}")
        st.write('Min battery charge (kWh):', f"{_np.round(_np.min(battery_levels), 3)}")
    except Exception as e:
        st.info(f"Could not run battery simulation: {e}")

    # Feature importance
    try:
        fi = model.feature_importances_
        feat_names = X.columns.tolist()
        fi_pairs = sorted(zip(feat_names, fi), key=lambda x: x[1], reverse=True)
        names, vals = zip(*fi_pairs)
        st.subheader("Feature Importance")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(names, vals)
        ax2.invert_yaxis()
        st.pyplot(fig2)
    except Exception:
        st.info("Could not compute feature importance for this model.")
else:
    st.info("No model available. Train a model using the sidebar button.")

st.markdown("---")
st.write("Usage: run `streamlit run streamlit_app.py` in this project directory.")
