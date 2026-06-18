import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.title("🍷 Wine Quality Prediction")

st.write("Enter wine parameters")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

df = pd.read_csv("winequality.csv")

df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]

model = DecisionTreeClassifier(max_depth=5, criterion="entropy")
model.fit(X, y)

if st.button("Predict Wine Quality"):

    sample = np.array([[fixed_acidity,
                        volatile_acidity,
                        citric_acid,
                        residual_sugar,
                        chlorides,
                        free_sulfur_dioxide,
                        total_sulfur_dioxide,
                        density,
                        pH,
                        sulphates,
                        alcohol]])

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.success("✅ Good Quality Wine")
    else:
        st.error("❌ Low Quality Wine")
