import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import requests
import streamlit as st

from services.api import predict_churn
from components.inputs import render_customer_inputs
from components.prediction import render_prediction
from components.analytics import render_customer_analytics
from components.model_insights import render_model_insights


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)


st.title("Customer Churn Intelligence")
st.caption(
    "Behavior-based customer churn intelligence powered by machine learning."
)


prediction_tab, analytics_tab, model_tab = st.tabs(
    [
        "Prediction",
        "Customer Analytics",
        "Model Insights",
    ]
)


with prediction_tab:

    features = render_customer_inputs()

    if st.button(
        "Predict Churn",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Analyzing customer behavior..."):
            try:
                result = predict_churn(features)

                st.divider()

                render_prediction(result)

            except requests.RequestException as exc:
                st.error(
                    f"Unable to connect to the prediction API: {exc}"
                )


with analytics_tab:
    render_customer_analytics()


with model_tab:
    render_model_insights()