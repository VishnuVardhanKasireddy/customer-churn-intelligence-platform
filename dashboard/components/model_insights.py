import plotly.express as px
import streamlit as st

from src.model.importance import get_saved_model_importance
from src.model.load_model import load_model


def render_model_insights():
    st.subheader("Model Insights")

    artifact = load_model()

    importance = get_saved_model_importance()

    threshold = artifact["threshold"]

    st.markdown("### Classification Threshold")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Classification Threshold",
            f"{threshold:.2f}",
        )

    with col2:
        st.info(
            f"Customers with a predicted churn probability "
            f"of **{threshold:.0%} or higher** are classified as churn."
        )

    st.divider()

    st.markdown("### Feature Importance")

    top_features = importance.head(10).sort_values(
        by="importance",
        ascending=True,
    )

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Random Forest Features",
        labels={
            "importance": "Importance",
            "feature": "Feature",
        },
    )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Feature importance indicates how much a feature contributes "
        "to the Random Forest's decision process. It does not establish "
        "causation."
    )