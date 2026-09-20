import streamlit as st


def render_customer_inputs():
    st.subheader("Customer Behavioral Profile")

    with st.expander("Customer Profile", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            total_orders = st.number_input(
                "Total Orders",
                min_value=0,
                value=5,
                step=1,
                help="Total number of orders during the observation window.",
            )

            total_spend = st.number_input(
                "Total Spend",
                min_value=0.0,
                value=1200.0,
                step=10.0,
                help="Total customer spending during the observation window.",
            )

        with col2:
            total_quantity = st.number_input(
                "Total Quantity",
                min_value=0,
                value=80,
                step=1,
                help="Total number of units purchased.",
            )

            unique_products = st.number_input(
                "Unique Products",
                min_value=0,
                value=15,
                step=1,
                help="Number of distinct products purchased.",
            )

        with col3:
            active_days = st.number_input(
                "Active Days",
                min_value=0,
                value=10,
                step=1,
                help="Number of distinct days on which the customer purchased.",
            )

            customer_lifespan_days = st.number_input(
                "Customer Lifespan",
                min_value=0,
                value=120,
                step=1,
                help="Days between the customer's first and most recent purchase.",
            )

    with st.expander("Purchase Behavior", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            average_order_value = st.number_input(
                "Average Order Value",
                min_value=0.0,
                value=240.0,
                step=10.0,
            )

        with col2:
            recency_days = st.number_input(
                "Recency",
                min_value=0,
                value=20,
                step=1,
                help="Days since the customer's most recent purchase.",
            )

        with col3:
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

    with st.expander("Recent Activity", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            orders_30d = st.number_input(
                "Orders — Last 30 Days",
                min_value=0,
                value=2,
                step=1,
            )

            spend_30d = st.number_input(
                "Spend — Last 30 Days",
                min_value=0.0,
                value=400.0,
                step=10.0,
            )

        with col2:
            orders_60d = st.number_input(
                "Orders — Last 60 Days",
                min_value=0,
                value=3,
                step=1,
            )

            spend_60d = st.number_input(
                "Spend — Last 60 Days",
                min_value=0.0,
                value=650.0,
                step=10.0,
            )

        with col3:
            orders_90d = st.number_input(
                "Orders — Last 90 Days",
                min_value=0,
                value=4,
                step=1,
            )

            spend_90d = st.number_input(
                "Spend — Last 90 Days",
                min_value=0.0,
                value=900.0,
                step=10.0,
            )

    with st.expander("Purchase Trend", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            orders_previous_90d = st.number_input(
                "Orders — Previous 90 Days",
                min_value=0,
                value=3,
                step=1,
            )

            order_trend_ratio = st.number_input(
                "Order Trend Ratio",
                min_value=0.0,
                value=1.33,
                step=0.01,
                help="Recent 90-day orders divided by previous 90-day orders.",
            )

        with col2:
            spend_previous_90d = st.number_input(
                "Spend — Previous 90 Days",
                min_value=0.0,
                value=700.0,
                step=10.0,
            )

            spend_trend_ratio = st.number_input(
                "Spend Trend Ratio",
                min_value=0.0,
                value=1.29,
                step=0.01,
                help="Recent 90-day spending divided by previous 90-day spending.",
            )

    return {
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