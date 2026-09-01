import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Bank Customer Rewards Dashboard",
    page_icon="💳",
    layout="wide",
)

DATA_PATH = Path("outputs") / "reward_segmented_customers.csv"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Data file not found. Please generate the dataset and reward segmentation first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

# Clean display values
if "Reward_Tier" in df.columns:
    tier_order = ["Platinum", "Gold", "Silver"]
    df["Reward_Tier"] = pd.Categorical(df["Reward_Tier"], categories=tier_order, ordered=True)

st.title("💳 Bank Customer Rewards Dashboard")
st.caption("Reward and discount segmentation for credit-card customers")

# Sidebar filters for non-technical users
st.sidebar.header("Filter Customers")
selected_tier = st.sidebar.multiselect(
    "Reward Tier",
    options=sorted(df["Reward_Tier"].dropna().unique().tolist()),
    default=sorted(df["Reward_Tier"].dropna().unique().tolist()),
)
selected_region = st.sidebar.multiselect(
    "Region",
    options=sorted(df["Region"].dropna().unique().tolist()),
    default=sorted(df["Region"].dropna().unique().tolist()),
)
selected_card = st.sidebar.multiselect(
    "Card Type",
    options=sorted(df["Card_Type"].dropna().unique().tolist()),
    default=sorted(df["Card_Type"].dropna().unique().tolist()),
)

filtered_df = df.copy()
if selected_tier:
    filtered_df = filtered_df[filtered_df["Reward_Tier"].isin(selected_tier)]
if selected_region:
    filtered_df = filtered_df[filtered_df["Region"].isin(selected_region)]
if selected_card:
    filtered_df = filtered_df[filtered_df["Card_Type"].isin(selected_card)]

# KPI metrics
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", f"{len(filtered_df):,}")
with col2:
    st.metric("Platinum Customers", int((filtered_df["Reward_Tier"] == "Platinum").sum()))
with col3:
    st.metric("Average Monthly Spend", f"${filtered_df['Monthly_Spend'].mean():,.0f}")
with col4:
    st.metric("Average Transactions", f"{filtered_df['Total_Transactions'].mean():,.0f}")

# Charts
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    tier_counts = filtered_df["Reward_Tier"].value_counts().reindex(["Platinum", "Gold", "Silver"], fill_value=0)
    st.bar_chart(tier_counts)

with chart_col2:
    if not filtered_df.empty:
        scatter = px.scatter(
            filtered_df,
            x="Monthly_Spend",
            y="Total_Transactions",
            color="Reward_Tier",
            size="Offers_Redeemed",
            title="Customer Value vs Transactions",
            hover_data=["Customer_ID", "Region", "Card_Type"],
            color_discrete_map={
                "Platinum": "#7D3C98",
                "Gold": "#2E86C1",
                "Silver": "#28B463",
            },
        )
        st.plotly_chart(scatter, use_container_width=True)

# Reward strategy explanation
st.subheader("Reward Strategy")

strategy = {
    "Platinum": "High-value spenders: premium rewards, highest cashback, priority offers",
    "Gold": "Strong spenders: standard rewards, seasonal discounts, moderate loyalty perks",
    "Silver": "Occasional users: basic offers, re-engagement campaigns, welcome discounts",
}

for tier in ["Platinum", "Gold", "Silver"]:
    if tier in filtered_df["Reward_Tier"].unique():
        st.info(f"**{tier}:** {strategy.get(tier, '')}")

# Customer table
st.subheader("Customer List")
customer_table = filtered_df[[
    "Customer_ID",
    "Reward_Tier",
    "Monthly_Spend",
    "Total_Transactions",
    "Offers_Redeemed",
    "Region",
    "Card_Type",
    "Usage_Level",
]].sort_values(["Reward_Tier", "Monthly_Spend"], ascending=[False, False])

st.dataframe(customer_table, use_container_width=True)

# Optional: summary by segment
st.subheader("Reward Segment Summary")
summary = (
    filtered_df.groupby("Reward_Tier", observed=False)
    .agg(
        Average_Monthly_Spend=("Monthly_Spend", "mean"),
        Average_Transactions=("Total_Transactions", "mean"),
        Average_Offers_Redeemed=("Offers_Redeemed", "mean"),
        Customer_Count=("Customer_ID", "count"),
    )
    .reset_index()
)
summary = summary.sort_values("Average_Monthly_Spend", ascending=False)
st.dataframe(summary, use_container_width=True)
