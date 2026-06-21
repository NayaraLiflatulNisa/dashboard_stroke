import streamlit as st

st.set_page_config(
    page_title="Dashboard Prediksi Stroke",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Dashboard Prediksi Stroke")

st.markdown("""
### Tentang Project

Dashboard ini dibuat untuk memprediksi kemungkinan seseorang mengalami stroke menggunakan algoritma **Decision Tree**.

### Fitur Dashboard

📊 Exploratory Data Analysis (EDA)

🌳 Visualisasi Model Decision Tree

📈 Evaluasi Model
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

🩺 Simulasi Prediksi Stroke

---

👈 Pilih menu di sidebar untuk melihat setiap halaman.
""")

st.image(
    "stroke.jpg",
    width=500
)