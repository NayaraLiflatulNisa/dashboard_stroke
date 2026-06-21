import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis (EDA)")

# Membaca dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Hapus kolom id
df = df.drop(columns=['id'])

# Isi nilai BMI yang kosong
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Statistik dataset
jumlah_data = len(df)
jumlah_fitur = len(df.columns)
jumlah_stroke = df["stroke"].sum()
jumlah_tidak_stroke = jumlah_data - jumlah_stroke

# Membuat kartu statistik
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Data", jumlah_data)
col2.metric("Jumlah Fitur", jumlah_fitur)
col3.metric("Stroke", jumlah_stroke)
col4.metric("Tidak Stroke", jumlah_tidak_stroke)

# Preview Dataset
st.subheader("Preview Dataset")
st.dataframe(df.head())

# Pie Chart Stroke
st.subheader("Distribusi Stroke")

stroke_count = df['stroke'].value_counts()

fig, ax = plt.subplots()

ax.pie(
    stroke_count,
    labels=["Tidak Stroke", "Stroke"],
    autopct='%1.1f%%'
)

st.pyplot(fig)

# Histogram Umur
st.subheader("Distribusi Umur")

fig, ax = plt.subplots()

ax.hist(df['age'], bins=20)

ax.set_xlabel("Umur")
ax.set_ylabel("Jumlah Pasien")

st.pyplot(fig)

# Informasi Dataset
st.subheader("Informasi Dataset")

st.write("""
Dataset Stroke digunakan untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan beberapa faktor kesehatan dan demografi.

Fitur yang digunakan antara lain:

- Gender
- Age (umur)
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

Target:
- Stroke = 1 → Mengalami Stroke
- Stroke = 0 → Tidak Mengalami Stroke
""")