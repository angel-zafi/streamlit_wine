import streamlit as st
import pandas as pd
import numpy as np

# page config
st.set_page_config(
    page_title = "Chocolate Sales Revenue Prediction",
    layout = "centered"
)

st.title("Chocolate Sales Revenue Prediction")
st.caption("Predict sales amount based on product, country, sales person, and shipment volume.")

# data
@st.cache_data
def load_data():
    df = pd.read_csv("Chocolate Sales (2).csv")

    # parse date
    df["Date"] = pd.to_datetime(df["Date"], errors = "coerce")

    # clean Amount
    if df["Amount"].dtype == "object":
        df["Amount"] = (
            df["Amount"]
            .str.replace("$", "", regex  =False)
            .str.replace(",", "", regex = False)
            .astype(float)
        )

    # extract simple date feature
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # drop rows with missing essential fields
    df = df.dropna(subset = ["Amount", "Boxes Shipped", "Month", "Year", "Sales Person", "Country", "Product"])

    return df

df = load_data()



# train 
@st.cache_resource
def train_model(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer

    target = "Amount"
    X = df.drop(columns = [target])
    y = df[target]

    # define columns
    cat_cols = ["Sales Person", "Country", "Product"]
    num_cols = ["Boxes Shipped", "Month", "Year"]

    # preprocessing
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators = 300,
        random_state = 42,
        n_jobs = -1
    )
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X[cat_cols + num_cols],
        y,
        test_size = 0.2,
        random_state = 42
    )

    # evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "rows": int(len(df)),
    }

    # stats for sliders
    stats = {}
    for col in num_cols:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    # options for dropdowns
    options = {
        "Sales Person": sorted(df["Sales Person"].dropna().unique().tolist()),
        "Country": sorted(df["Country"].dropna().unique().tolist()),
        "Product": sorted(df["Product"].dropna().unique().tolist()),
    }

    return pipeline, cat_cols, num_cols, metrics, stats, options

pipeline, cat_cols, num_cols, metrics, stats, options = train_model(df)



# sidebar thing
with st.sidebar:
    show_data = st.toggle("Show dataset preview", False)
    show_metrics = st.toggle("Show model metrics", True)



# user input
st.subheader("Inputs")

sales_person = st.selectbox("Sales Person", options["Sales Person"])
country = st.selectbox("Country", options["Country"])
product = st.selectbox("Product", options["Product"])

boxes_stats = stats["Boxes Shipped"]
boxes_shipped = st.number_input(
    "Boxes Shipped",
    min_value = int(boxes_stats["min"]),
    max_value = int(boxes_stats["max"]),
    value = int(boxes_stats["median"]),
    step = 1
)

month_stats = stats["Month"]
month = st.slider(
    "Month",
    min_value = int(month_stats["min"]),
    max_value = int(month_stats["max"]),
    value = int(month_stats["median"]),
    step = 1
)

year_stats = stats["Year"]
year = st.slider(
    "Year",
    min_value = int(year_stats["min"]),
    max_value = int(year_stats["max"]),
    value = int(year_stats["median"]),
    step = 1
)



# predictt
if st.button("Predict Sales Amount"):
    input_df = pd.DataFrame([{
        "Sales Person": sales_person,
        "Country": country,
        "Product": product,
        "Boxes Shipped": boxes_shipped,
        "Month": month,
        "Year": year
    }])

    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted Sales Amount: ${prediction:,.2f}")



# outpusts
if show_metrics:
    st.subheader("Model Metrics")
    st.write(f"Mean Absolute Error: {metrics['mae']:.2f}")
    st.write(f"R² Score: {metrics['r2']:.3f}")
    st.write(f"Rows used: {metrics['rows']}")

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))




# the styling of steamlit
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            180deg,
            rgba(30,30,30,0.20),
            rgba(0,0,0,0.35)
        );
    }
    </style>
    """,
    unsafe_allow_html = True
)
