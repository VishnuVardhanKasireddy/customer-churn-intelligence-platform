import streamlit as st

from services.api import predict_churn


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)


st.title("Customer Churn Intelligence")
st.write(
    "Predict customer churn using behavioral transaction features."
)


st.header("Customer Behavioral Features")


col1, col2, col3 = st.columns(3)


with col1:
    total_orders = st.number_input(
        "Total Orders",
        min_value=0,
        value=5,
        step=1,
    )

    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        value=1200.0,
        step=10.0,
    )

    total_quantity = st.number_input(
        "Total Quantity",
        min_value=0,
        value=80,
        step=1,
    )

    unique_products = st.number_input(
        "Unique Products",
        min_value=0,
        value=15,
        step=1,
    )

    active_days = st.number_input(
        "Active Days",
        min_value=0,
        value=10,
        step=1,
    )

    average_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0,
        value=240.0,
        step=10.0,
    )


with col2:
    customer_lifespan_days = st.number_input(
        "Customer Lifespan (days)",
        min_value=0,
        value=120,
        step=1,
    )

    recency_days = st.number_input(
        "Recency (days)",
        min_value=0,
        value=20,
        step=1,
    )

    average_interpurchase_days = st.number_input(
        "Average Interpurchase Days",
        min_value=0.0,
        value=25.0,
        step=1.0,
    )

    median_interpurchase_days = st.number_input(
        "Median Interpurchase Days",
        min_value=0.0,
        value=22.0,
        step=1.0,
    )

    orders_30d = st.number_input(
        "Orders (30 days)",
        min_value=0,
        value=2,
        step=1,
    )

    orders_60d = st.number_input(
        "Orders (60 days)",
        min_value=0,
        value=3,
        step=1,
    )

    orders_90d = st.number_input(
        "Orders (90 days)",
        min_value=0,
        value=4,
        step=1,
    )


with col3:
    spend_30d = st.number_input(
        "Spend (30 days)",
        min_value=0.0,
        value=400.0,
        step=10.0,
    )

    spend_60d = st.number_input(
        "Spend (60 days)",
        min_value=0.0,
        value=650.0,
        step=10.0,
    )

    spend_90d = st.number_input(
        "Spend (90 days)",
        min_value=0.0,
        value=900.0,
        step=10.0,
    )

    orders_previous_90d = st.number_input(
        "Orders (Previous 90 days)",
        min_value=0,
        value=3,
        step=1,
    )

    spend_previous_90d = st.number_input(
        "Spend (Previous 90 days)",
        min_value=0.0,
        value=700.0,
        step=10.0,
    )

    order_trend_ratio = st.number_input(
        "Order Trend Ratio",
        min_value=0.0,
        value=1.33,
        step=0.01,
    )

    spend_trend_ratio = st.number_input(
        "Spend Trend Ratio",
        min_value=0.0,
        value=1.29,
        step=0.01,
    )


if st.button("Predict Churn", type="primary"):

    features = {
        "total_orders": total_orders,
        "total_spend": total_spend,
        "total_quantity": total_quantity,
        "unique_products": unique_products,
        "active_days": active_days,
        "average_order_value": average_order_value,
        "customer_lifespan_days": customer_lifespan_days,
        "recency_days": recency_days,
        "average_interpurchase_days": average_interpurchase_days,
        "median_interpurchase_days": median_interpurchase_days,
        "orders_30d": orders_30d,
        "orders_60d": orders_60d,
        "orders_90d": orders_90d,
        "spend_30d": spend_30d,
        "spend_60d": spend_60d,
        "spend_90d": spend_90d,
        "orders_previous_90d": orders_previous_90d,
        "spend_previous_90d": spend_previous_90d,
        "order_trend_ratio": order_trend_ratio,
        "spend_trend_ratio": spend_trend_ratio,
    }

    try:
        result = predict_churn(features)

        st.header("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{result['churn_probability']:.2%}",
        )

        col1, col2 = st.columns(2)

        with col1:
            prediction = result["churn_prediction"]

            if prediction == 1:
                st.error("Customer predicted to churn")
            else:
                st.success("Customer predicted to remain active")

        with col2:
            st.info(
                f"Risk Level: {result['risk_level'].upper()}"
            )

    except requests.RequestException:
        st.error(
            "Unable to connect to the prediction API. "
            "Make sure FastAPI is running."
        )