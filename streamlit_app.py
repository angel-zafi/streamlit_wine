import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Chocolate Sales Revenue Prediction", layout = "centered")
st.title("Chocolate Sales Revenue Prediction")
st.caption("Predict expected sales amount based on product, country, sales person, shipment quantity, and time period")

@st.cache_data
def load_data():
    df = pd.read_csv("Chocolate Sales (2).csv")

    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace("$", "", regex = False)
        .str.replace(",", "", regex = False)
        .str.strip()
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors = "coerce")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, errors = "coerce")
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df = df.dropna(subset = ["Amount", "Boxes Shipped", "Month", "Year", "Sales Person", "Country", "Product"])
    df = df.drop_duplicates()

    return df

@st.cache_resource
def load_model():
    return joblib.load("chocolate_sales_model.pkl")

df = load_data()
model = load_model()

with st.sidebar:
    show_data = st.toggle("Show dataset preview", False)

st.subheader("Inputs")

sales_person = st.selectbox("Sales Person", sorted(df["Sales Person"].unique()))
country = st.selectbox("Country", sorted(df["Country"].unique()))
product = st.selectbox("Product", sorted(df["Product"].unique()))

boxes_shipped = st.number_input(
    "Boxes Shipped",
    min_value = int(df["Boxes Shipped"].min()),
    max_value = int(df["Boxes Shipped"].max()),
    value = int(df["Boxes Shipped"].median()),
    step = 1
)

month = st.slider("Month", 1, 12, int(df["Month"].median()))
year = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].median()))

if st.button("Predict Sales Amount"):
    input_df = pd.DataFrame([{
        "Sales Person": sales_person,
        "Country": country,
        "Product": product,
        "Boxes Shipped": boxes_shipped,
        "Month": month,
        "Year": year
    }])

    pred = model.predict(input_df)[0]
    st.success(f"Predicted Sales Amount: ${pred:,.2f}")

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
