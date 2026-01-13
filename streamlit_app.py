import joblib
import streamlit as st
import pandas as pd
import numpy as np

# page config
st.set_page_config(
    page_title="Wine Quality Prediction",
    layout="centered"
)

st.title("Wine Quality Prediction")
st.caption("Predict wine quality using red and white wine datasets.")



# data
@st.cache_data
def load_data():
    red = pd.read_csv("winequality-red.csv", sep = ";")
    white = pd.read_csv("winequality-white.csv", sep = ";")

    red["wine_type"] = "red"
    white["wine_type"] = "white"

    return pd.concat([red, white], ignore_index=True)

df = load_data()



#train
@st.cache_resource
def train_model(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor

    target = "quality"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = ["wine_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers = [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", allow := preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "rows": len(df)
    }

    stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median())
        }
        for col in num_cols
    }

    return pipeline, num_cols, metrics, stats

pipeline, num_cols, metrics, stats = train_model(df)



# sidebar thing
with st.sidebar:
    show_data = st.toggle("Show dataset preview", False)
    show_metrics = st.toggle("Show model metrics", True)


# user inputs 
wine_type = st.selectbox("Wine Type", ["red", "white"])

st.subheader("Chemical Properties")

inputs = {}
for col in num_cols:
    s = stats[col]
    rng = s["max"] - s["min"]

    if rng <= 1:
        step = 0.01
    elif rng <= 10:
        step = 0.1
    else:
        step = 0.5

    inputs[col] = st.slider(
        col,
        min_value = s["min"],
        max_value = s["max"],
        value = s["median"],
        step = step
    )



# predict
if st.button("Predict Wine Quality"):
    input_df = pd.DataFrame([{**inputs, "wine_type": wine_type}])
    prediction = pipeline.predict(input_df)[0]

    st.success(f"Predicted quality (raw): {prediction:.2f}")
    st.info(f"Predicted quality (rounded): {int(np.round(prediction))}")





if show_metrics:
    st.subheader("Model Metrics")
    st.write(f"Mean Absolute Error: {metrics['mae']:.3f}")
    st.write(f"R² Score: {metrics['r2']:.3f}")
    st.write(f"Rows used: {metrics['rows']}")

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))



# page thing
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            180deg,
            rgba(30,30,30,0.35),
            rgba(0,0,0,0.55)
        ),
    }
    </style>
    """,
    unsafe_allow_html = True
)