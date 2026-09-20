import plotly.express as px


def create_churn_distribution(churn_counts):
    fig = px.bar(
        churn_counts,
        x="status",
        y="customers",
        text="customers",
        labels={
            "status": "Customer Status",
            "customers": "Snapshots",
        },
        title="Customer Snapshot Churn Distribution",
    )

    fig.update_traces(
        textposition="outside",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_recency_boxplot(data):
    fig = px.box(
        data,
        x="status",
        y="recency_days",
        points=False,
        labels={
            "status": "Customer Status",
            "recency_days": "Recency (days)",
        },
        title="Recency Distribution by Churn Status",
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_spend_distribution(data):
    fig = px.histogram(
        data,
        x="total_spend",
        nbins=50,
        labels={
            "total_spend": "Total Spend",
        },
        title="Customer Spend Distribution",
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_orders_vs_spend_scatter(data):
    fig = px.scatter(
        data,
        x="total_orders",
        y="total_spend",
        color="status",
        opacity=0.6,
        hover_data=[
            "recency_days",
            "average_order_value",
            "unique_products",
        ],
        labels={
            "total_orders": "Total Orders",
            "total_spend": "Total Spend",
            "status": "Customer Status",
        },
        title="Customer Orders vs Spending",
    )

    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_monthly_churn_trend(monthly_data):
    fig = px.line(
        monthly_data,
        x="cutoff_date",
        y="churn_rate",
        markers=True,
        labels={
            "cutoff_date": "Snapshot Date",
            "churn_rate": "Churn Rate",
        },
        title="Churn Rate by Snapshot Date",
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_tickformat=".0%",
    )

    return fig