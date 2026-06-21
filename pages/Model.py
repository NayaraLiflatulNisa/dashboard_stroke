import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.title("🌳 Model Decision Tree")

# Membaca dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Hapus kolom id
df = df.drop(columns=['id'])

# Isi BMI yang kosong
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Pisahkan fitur dan target
X = df_encoded.drop(columns=['stroke'])
y = df_encoded['stroke']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Decision Tree
model_viz = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    random_state=42
)

model_viz.fit(X_train, y_train)

# Prediksi
y_pred = model_viz.predict(X_test)

# Metric
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("📊 Top 10 Feature Importance")

fitur = X.columns
importance = model_viz.feature_importances_

imp_df = pd.DataFrame({
    'fitur': fitur,
    'importance': importance
})

imp_df = imp_df.sort_values(
    by='importance',
    ascending=False
).head(10)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    imp_df['fitur'][::-1],
    imp_df['importance'][::-1]
)

ax.set_xlabel("Tingkat Kepentingan")
ax.set_ylabel("Fitur")

st.pyplot(fig)

st.subheader("🌳 Visualisasi Decision Tree")

fig, ax = plt.subplots(figsize=(25,10))

plot_tree(
    model_viz,
    feature_names=X.columns,
    class_names=['Tidak Stroke', 'Stroke'],
    filled=True,
    rounded=True,
    fontsize=6,
    ax=ax
)

st.pyplot(fig)

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Tidak Stroke', 'Stroke'],
    yticklabels=['Tidak Stroke', 'Stroke']
)

ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")

st.pyplot(fig)

st.subheader("📈 Performa Model")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

c2.metric(
    "Precision",
    f"{precision:.2%}"
)

c3.metric(
    "Recall",
    f"{recall:.2%}"
)

c4.metric(
    "F1-Score",
    f"{f1:.2%}"
)

st.info(f"""
Model Decision Tree memiliki:

- Accuracy : {accuracy:.2%}
- Precision : {precision:.2%}
- Recall : {recall:.2%}
- F1-Score : {f1:.2%}

Dataset stroke bersifat tidak seimbang (imbalanced),
karena jumlah pasien stroke jauh lebih sedikit dibandingkan
pasien tidak stroke.
""")