import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.title("🩺 Simulasi Prediksi Stroke")

# Membaca dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Hapus kolom id
df = df.drop(columns=['id'])

# Isi BMI kosong
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Encoding
df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

# Pisahkan fitur dan target
X = df_encoded.drop(columns=['stroke'])
y = df_encoded['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model_viz = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    random_state=42
)

model_viz.fit(X_train, y_train)

umur = st.slider(
    "Umur",
    0,
    100,
    30
)

hipertensi = st.selectbox(
    "Hipertensi",
    ["Tidak", "Ya"]
)

penyakit_jantung = st.selectbox(
    "Penyakit Jantung",
    ["Tidak", "Ya"]
)

glukosa = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    value=25.0
)

prediksi_btn = st.button(
    "Prediksi"
)

if prediksi_btn:

    data_baru = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )

    data_baru['age'] = umur

    data_baru['hypertension'] = (
        1 if hipertensi == "Ya"
        else 0
    )

    data_baru['heart_disease'] = (
        1 if penyakit_jantung == "Ya"
        else 0
    )

    data_baru['avg_glucose_level'] = glukosa

    data_baru['bmi'] = bmi

    hasil = model_viz.predict(data_baru)

    prob = model_viz.predict_proba(
        data_baru
    )

    st.subheader(
        "Probabilitas Prediksi"
    )

    st.write(
        f"Tidak Stroke : {prob[0][0]*100:.2f}%"
    )

    st.write(
        f"Stroke : {prob[0][1]*100:.2f}%"
    )

    if hasil[0] == 1:

        st.error(
            "🔴 Hasil Prediksi : Berisiko Stroke"
        )

    else:

        st.success(
            "🟢 Hasil Prediksi : Tidak Berisiko Stroke"
        )