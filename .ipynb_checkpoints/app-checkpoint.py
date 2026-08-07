import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
st.markdown("""
<style>

/* Ocean Blue canvas background */
[data-testid="stAppViewContainer"] {
    background-color: #0077B6;  /* Ocean Blue */
}

/* Remove top white header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Main title in ivory white */
h1 {
    color: #FFFFF0 !important;   /* Ivory White */
    text-align: center;
    font-weight: bold;
}

/* Subheadings */
h2, h3, h4 {
    color: #86EFAC !important;   /* Light Green */
}

/* Labels / Input text */
label {
    color: #D1FAE5 !important;   /* Soft Green */
}

/* Buttons */
.stButton > button {
    background-color: #22C55E;
    color: white;
    border-radius: 12px;
}
.stButton > button:hover {
    background-color: #16A34A;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------LOAD FILES-------------------------------------
model = joblib.load("best_model_final_svr.pkl")
scaler = joblib.load("scaler.pkl")
feature_info = joblib.load("feature_columns.pkl")
feature_columns = feature_info['all_columns'] 
numeric_cols = feature_info['numeric_cols']
categorical_cols = feature_info['categorical_cols']
emission_range = joblib.load("emission_range.pkl")
EMISSION_MIN = float(emission_range["min"])
EMISSION_MAX = float(emission_range["max"])
# ----------------------------------PAGE CONFIG------------------------------------
st.set_page_config(page_title="Carbon Emission Prediction",layout="centered")
st.title("🌏🍃 ECO-PREDICT :")
st.title("Carbon Emission Prediction System")
st.caption("Enter model and infrastructure details to predict carbon emission (kg CO₂)")
# ----------------------------------USER INPUTS-------------------------------------
st.subheader("🧠 Model Configuration")
model_type = st.selectbox("Model type",["CNN", "LLM", "Diffusion", "Transformer"])
task_type = st.selectbox("Task type",["detection","generation","classification","segmentation"])
training_framework = st.selectbox("Training framework",["TensorFlow", "PyTorch", "JAX"])
gpu_type = st.selectbox("GPU Type",["RTX3090", "V100", "T4", "A100"])
cloud_provider = st.selectbox("Cloud Provider",["AWS", "Azure", "GCP", "On-Prem"])
year = st.number_input("📅 Year",min_value=2000,max_value=2050,value=2024,step=1)

st.subheader("⚙️Training Parameters")
epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32)
training_time = st.number_input("Training Time (hours)", min_value=0.1, value=10.0)
model_size = st.number_input("Model size(million params)", min_value=1.0, max_value=5000.0, value=200.0)
learning_rate = st.number_input("Learning Rate",min_value=0.00001, max_value=1.0, value=0.001, format="%.5f")
final_train_loss = st.number_input("Final Training Loss",min_value=0.0, max_value=10.0, value=0.25, format="%.5f" )
final_val_loss = st.number_input("Final Validation Loss",min_value=0.0, max_value=10.0, value=0.30, format="%.5f")
overfitting_score = st.number_input("Overfitting Score",min_value=0.0, max_value=5.0, value=0.05, format="%.5f")

st.subheader("📊Performance & Efficiency")
inference_latency_ms = st.number_input("Inference Latency (ms)",min_value=1.0, max_value=5000.0, value=50.0)
throughput_samples_per_sec = st.number_input("Throughput (samples/sec)",min_value=1.0, max_value=10000.0, value=120.0)
gpu_memory_gb = st.number_input("GPU Memory Usage (GB)",min_value=2, max_value=80, value=8)

st.subheader("🎯Evaluation Metrics")
model_accuracy = st.slider("Model Accuracy", 0.0, 1.0, 0.85, step=0.001, format="%.3f")
f1_score = st.slider("F1 Score", 0.0, 1.0, 0.82, step=0.001, format="%.3f")
roc_auc = st.slider("ROC AUC", 0.0, 1.0, 0.88, step=0.001, format="%.3f")

input_data = {
    "year":year,
    "model_size_million_params": model_size,
    "learning_rate": learning_rate,
    "final_train_loss": final_train_loss,
    "final_val_loss": final_val_loss,
    "overfitting_score": overfitting_score,
    "inference_latency_ms": inference_latency_ms,
    "throughput_samples_per_sec": throughput_samples_per_sec,
    "model_accuracy": model_accuracy,
    "f1_score": f1_score,
    "roc_auc": roc_auc,
    "gpu_memory_gb": gpu_memory_gb,

    "epochs": epochs,
    "batch_size": batch_size,
    "training_time_hours": training_time,

    "model_type": model_type,
    "task_type": task_type,
    "training_framework": training_framework,
    "gpu_type": gpu_type,
    "cloud_provider": cloud_provider
}


input_df = pd.DataFrame([input_data])

#------------- One-hot encode & align features ----------------------------------
input_df_encoded = pd.get_dummies(input_df)

#------------ Add missing columns from training----------------------------------
for col in feature_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

#------------Drop extra columns not in training (safety) ------------------------
input_df_encoded = input_df_encoded[feature_columns]

#--------------------- Scale numeric columns ------------------------------------
input_df_encoded[numeric_cols] = scaler.transform(input_df_encoded[numeric_cols])

if st.button("🌱 Predict Carbon Emission"):

    # ---------- MODEL PREDICTION ----------
    # debug - checking encoding and scaling working or not
    st.write(input_df_encoded)
    raw_emission = float(model.predict(input_df_encoded).ravel()[0])

    # --------- DATA-DRIVEN RISK THRESHOLDS ----------
    low_th = EMISSION_MIN + 0.33 * (EMISSION_MAX - EMISSION_MIN)
    mid_th = EMISSION_MIN + 0.66 * (EMISSION_MAX - EMISSION_MIN)

    if raw_emission <= low_th:
        risk = "Low"
        color = "#16A34A"
    elif raw_emission <= mid_th:
        risk = "Medium"
        color = "#F59E0B"
    else:
        risk = "High"
        color = "#DC2626"

    # --------- GAUGE NORMALIZATION ----------
    gauge_value = ((raw_emission - EMISSION_MIN) /
                   (EMISSION_MAX - EMISSION_MIN)) * 100

    gauge_value = round(max(0, min(gauge_value, 100)), 1)

    # --------- DISPLAY ----------
    st.success(f"🌿 Predicted Carbon Emission: {raw_emission:.2f} kg CO₂")

    st.markdown(
        f"""
        <div style="
            padding:12px;
            border-radius:8px;
            border-left:6px solid {color};
            font-size:22px;
            font-weight:600;
            color:{color};
        ">
            🌡 Risk Level: {risk}
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={'suffix': " %"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 33], 'color': "#86EFAC"},
                {'range': [33, 66], 'color': "#FCD34D"},
                {'range': [66, 100], 'color': "#FCA5A5"}
            ],
            'bar': {'color': color}
        }
    ))

    st.plotly_chart(fig,use_container_width=True)
    st.caption("Gauge shows relative emission level based on dataset range.")