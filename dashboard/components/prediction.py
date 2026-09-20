import plotly.graph_objects as go
import streamlit as st


def render_prediction(result):
    probability = result["churn_probability"]
    prediction = result["churn_prediction"]
    risk_level = result["risk_level"]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}",
        )

    with col2:
        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn",
        )

    with col3:
        st.metric(
            "Risk Level",
            risk_level.upper(),
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
            },
            title={"text": "Churn Probability"},
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if prediction == 1:
        st.error(
            "This customer is predicted to churn within the defined prediction window."
        )
    else:
        st.success(
            "This customer is predicted to remain active within the defined prediction window."
        )