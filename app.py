import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib

# --- Load Data --------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/raveendran2862/dataset/refs/heads/main/house_prices.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()
st.write("📊 Dataset Preview", df.head())

# --- Preprocessing ----------------------------------------------------
st.write("🚀 Preprocessing and training model...")


target = "price"

if target not in df.columns:
    st.error("❌ 'SalePrice' column not found – adjust target name.")
    st.stop()

# Clean & prep numeric-only first pass
num_df = df.select_dtypes(include=np.number).dropna(axis=1, how="any")
X = num_df.drop(columns=[target])
y = num_df[target]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
score = r2_score(y_test, model.predict(X_test_scaled))
st.write(f"✅ Model Trained — R² Score: **{score:.3f}**")

# Save model + scaler (optional)
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# --- User Input -------------------------------------------------------
st.sidebar.header("🏡 House Feature Inputs")

inputs = {}
for col in X.columns:
    # numeric slider range from dataset
    mn, mx = float(df[col].min()), float(df[col].max())
    step = (mx - mn) / 100
    default = float(df[col].median())
    inputs[col] = st.sidebar.slider(col, mn, mx, default, step=step)

# Build input array
input_array = np.array([list(inputs.values())])
scaled_input = scaler.transform(input_array)

# Predict
if st.sidebar.button("🔍 Predict Price"):
    pred = model.predict(scaled_input)[0]
    st.write(f"💰 **Predicted House Price:** ₹{pred:,.2f}")
