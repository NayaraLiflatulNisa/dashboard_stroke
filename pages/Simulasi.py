import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.title("🩺 Simulasi Prediksi Stroke")

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Hapus ID
df = df.drop(columns=["id"])

# Isi missing BMI
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# One Hot Encoding
df_encoded = pd.get_dummies(
    df,
    columns=[
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status"
    ],
    drop_first=True
)

X = df_encoded.drop(columns=["stroke"])
y = df_encoded["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Model Decision Tree
model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# INPUT USER
# ==========================

st.subheader("Masukkan Data Pasien")

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

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

jantung = st.selectbox(
    "Penyakit Jantung",
    ["Tidak", "Ya"]
)

menikah = st.selectbox(
    "Pernah Menikah",
    ["No", "Yes"]
)

work = st.selectbox(
    "Jenis Pekerjaan",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

residence = st.selectbox(
    "Tempat Tinggal",
    [
        "Urban",
        "Rural"
    ]
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

smoking = st.selectbox(
    "Status Merokok",
    [
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown"
    ]
)

# ==========================
# PREDIKSI
# ==========================

if st.button("Prediksi"):

    data_baru = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )

    # Fitur numerik
    data_baru["age"] = umur
    data_baru["hypertension"] = 1 if hipertensi == "Ya" else 0
    data_baru["heart_disease"] = 1 if jantung == "Ya" else 0
    data_baru["avg_glucose_level"] = glukosa
    data_baru["bmi"] = bmi

    # Gender
    if "gender_Male" in X.columns and gender == "Male":
        data_baru["gender_Male"] = 1

    if "gender_Other" in X.columns and gender == "Other":
        data_baru["gender_Other"] = 1

    # Ever Married
    if "ever_married_Yes" in X.columns and menikah == "Yes":
        data_baru["ever_married_Yes"] = 1

    # Work Type
    if f"work_type_{work}" in X.columns:
        data_baru[f"work_type_{work}"] = 1

    # Residence
    if "Residence_type_Urban" in X.columns and residence == "Urban":
        data_baru["Residence_type_Urban"] = 1

    # Smoking
    if smoking != "Unknown":
        kolom = f"smoking_status_{smoking}"
        if kolom in X.columns:
            data_baru[kolom] = 1

    # Prediksi
    hasil = model.predict(data_baru)
    probabilitas = model.predict_proba(data_baru)

    st.subheader("📊 Hasil Prediksi")

    st.write(
        f"**Probabilitas Tidak Stroke :** {probabilitas[0][0]*100:.2f}%"
    )

    st.write(
        f"**Probabilitas Stroke :** {probabilitas[0][1]*100:.2f}%"
    )

    if hasil[0] == 1:
        st.error("🔴 Pasien diprediksi berisiko Stroke")
    else:
        st.success("🟢 Pasien diprediksi Tidak Berisiko Stroke")