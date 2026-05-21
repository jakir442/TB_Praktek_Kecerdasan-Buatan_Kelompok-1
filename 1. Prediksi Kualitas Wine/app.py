import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("wine_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🍷 Prediksi Kualitas Wine")

# Input user
fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")

if st.button("Prediksi"):

    sulfur_ratio = (
        free_sulfur_dioxide /
        total_sulfur_dioxide
        if total_sulfur_dioxide != 0 else 0
    )

    acidity_ratio = (
        fixed_acidity /
        volatile_acidity
        if volatile_acidity != 0 else 0
    )

    data = pd.DataFrame([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        sulfur_ratio,
        acidity_ratio
    ]], columns=[
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
        'sulfur_ratio',
        'acidity_ratio'
    ])

    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)[0]

    hasil = le.inverse_transform([pred])[0]

    st.success(f"Prediksi Quality Wine: {hasil}")