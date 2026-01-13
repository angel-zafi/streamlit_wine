import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")
st.title("🍷 Wine Quality Prediction")
st.caption("Predict wine quality using red + white wine datasets.")

# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data
def load_data():
    # These CSVs use ; as separator (as shown in your sample)
    red = pd.read_csv("winequality-red.csv", sep=";")
    white = pd.read_csv("winequality-white.csv", sep=";")

    red["wine_type"] = "red"
    white["wine_type"] = "white"

    df = pd.concat([red, white], ignore_index=True)
    return df

df = load_data()

# ----------------------------
# Train model (cached)
# ----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "quality"
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = ["wine_type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
        "num_features": len(X.columns),
        "rows": len(df),
    }

    # For “nice” slider defaults + ranges
    stats = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in num_cols
    }

    return pipe, num_cols, metrics, stats

pipe, num_cols, metrics, stats = train_model(df)

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Options")
    show_data = st.toggle("Show dataset preview", value=False)
    show_metrics = st.toggle("Show model metrics", value=True)

# ----------------------------
# Main UI
# ----------------------------
wine_type = st.selectbox("Select Wine Type", ["red", "white"])

st.subheader("Enter chemical properties")

# Create inputs dynamically from numeric columns
input_values = {}
for col in num_cols:
    s = stats[col]
    # Make sliders feel sane even if range is huge:
    # use median as default, and clamp step sizes
    min_v, max_v, default_v = s["min"], s["max"], s["median"]

    # choose a reasonable step
    rng = max_v - min_v
    if rng <= 1:
        step = 0.01
    elif rng <= 10:
        step = 0.1
    else:
        step = 0.5

    input_values[col] = st.slider(
        label=col,
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(default_v),
        step=float(step),
    )

# Predict button
if st.button("Predict Wine Quality"):
    input_df = pd.DataFrame([{**input_values, "wine_type": wine_type}])
    pred = pipe.predict(input_df)[0]

    # quality is discrete in dataset; we can show both raw + rounded
    st.success(f"Predicted Quality (raw): {pred:.2f}")
    st.info(f"Predicted Quality (rounded): {int(np.round(pred))}")

# ----------------------------
# Extras
# ----------------------------
if show_metrics:
    st.subheader("📊 Model snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.3f}")
    c2.metric("R²", f"{metrics['R2']:.3f}")
    c3.metric("Rows used", f"{metrics['rows']}")

if show_data:
    st.subheader("🔎 Data preview")
    st.dataframe(df.head(10))

# ----------------------------
# Optional background styling (similar vibe to your example)
# ----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, rgba(30,30,30,0.35), rgba(0,0,0,0.55)),
                    url("https://images.unsplash.com/photo-1510626176961-4b57d4fbad03?auto=format&fit=crop&w=1600&q=60");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
