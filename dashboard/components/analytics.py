from pathlib import Path

import pandas as pd
import streamlit as st

from components.charts import (
    create_churn_distribution,
    create_monthly_churn_trend,
    create_orders_vs_spend_scatter,
    create_recency_boxplot,
    create_spend_distribution,
)


FEATURE_DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "customer_churn_features.csv"
)


@st.cache_data
def load_feature_data():
    df = pd.read_csv(FEATURE_DATA_FILE)

    df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

    return df


def render_customer_analytics():
    st.subheader("Customer Analytics")

    df = load_feature_data()

    total_customers = df["customer_id"].nunique()
    total_snapshots = len(df)
    churn_rate = df["churn"].mean()
    average_spend = df["total_spend"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            f"{total_customers:,}",
        )

    with col2:
        st.metric(
            "Customer Snapshots",
            f"{total_snapshots:,}",
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2%}",
        )

    with col4:
        st.metric(
            "Average Spend",
            f"{average_spend:,.2f}",
        )

    st.divider()

    
    # Churn distribution
    st.markdown("### Churn Distribution")

    churn_counts = (
        df["churn"]
        .map({0: "Retained", 1: "Churned"})
        .value_counts()
        .rename_axis("status")
        .reset_index(name="customers")
    )

    fig = create_churn_distribution(churn_counts)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Recency analysis
    st.markdown("### Recency and Churn")

    recency_df = df.copy()

    recency_df["status"] = recency_df["churn"].map(
        {
            0: "Retained",
            1: "Churned",
        }
    )

    fig = create_recency_boxplot(recency_df)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # Orders vs spend
    st.markdown("### Orders vs Spend")

    sample = df.sample(
        min(len(df), 5000),
        random_state=42,
    )

    sample["status"] = sample["churn"].map(
        {
            0: "Retained",
            1: "Churned",
        }
    )

    fig = create_orders_vs_spend_scatter(sample)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Monthly churn trend
    st.markdown("### Churn Trend Over Time")

    monthly_churn = (
        df.groupby("cutoff_date")["churn"]
        .mean()
        .reset_index()
        .rename(columns={"churn": "churn_rate"})
    )

    fig = create_monthly_churn_trend(monthly_churn)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Spend distribution
    st.markdown("### Customer Spending")

    fig = create_spend_distribution(df)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )