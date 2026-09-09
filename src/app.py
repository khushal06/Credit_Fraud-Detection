import joblib
import pandas as pd
import streamlit as st

try:
    from src.config import MODELS_DIR, TARGET
    from src.data import load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from config import MODELS_DIR, TARGET
    from data import load_dataset


st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")


@st.cache_data
def get_dataset():
    return load_dataset()


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODELS_DIR / "xgboost_model.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return model, feature_names, scaler


def score_transaction(row: pd.Series, feature_names, scaler, model) -> float:
    features = pd.DataFrame([row[feature_names]])
    features[["Time", "Amount"]] = scaler.transform(features[["Time", "Amount"]])
    return float(model.predict_proba(features)[0, 1])


st.title("Credit Card Fraud Detection Demo")

model, feature_names, scaler = load_model_assets()
full_df = get_dataset()
fraud_df = full_df[full_df[TARGET] == 1]
legit_df = full_df[full_df[TARGET] == 0]

threshold = st.slider("Fraud probability threshold", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
transaction_mode = st.radio("Choose transaction type", ["Random row", "Known fraud", "Known legit"])

if "sample_key" not in st.session_state:
    st.session_state.sample_key = None

if st.button("Draw new sample") or st.session_state.sample_key is None or st.session_state.get("last_mode") != transaction_mode:
    if transaction_mode == "Random row":
        source = full_df
    elif transaction_mode == "Known fraud":
        source = fraud_df
    else:
        source = legit_df

    if source.empty:
        st.warning("No matching rows found in the dataset.")
        st.stop()

    st.session_state.sample_key = source.sample(n=1).index[0]
    st.session_state.last_mode = transaction_mode

if transaction_mode == "Random row":
    source = full_df
elif transaction_mode == "Known fraud":
    source = fraud_df
else:
    source = legit_df

sample = source.loc[st.session_state.sample_key]
prob = score_transaction(sample, feature_names, scaler, model)
label = "Fraud" if prob >= threshold else "Not Fraud"

st.subheader("Selected transaction")
st.json(sample.to_dict())

st.subheader("Prediction")
col1, col2 = st.columns(2)
col1.metric("Fraud probability", f"{prob:.4f}")
col2.metric("Decision at threshold", label)
st.progress(prob)
