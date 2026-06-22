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
    f1_score,
    classification_report
)

st.title("🌳 Model Decision Tree")

# ======================
# Membaca Dataset
# ======================
df = pd.read_csv(
    "healthcare-dataset-stroke-data.csv",
    na_values=['N/A']
)

# Drop kolom id
df = df.drop(columns=['id'])

# Isi missing value bmi dengan median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# One-hot encoding
kategorikal_cols = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

df_encoded = pd.get_dummies(
    df,
    columns=kategorikal_cols,
    drop_first=True
)

# ======================
# Pisahkan fitur & target
# ======================
X = df_encoded.drop(columns=['stroke'])
y = df_encoded['stroke']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ======================
# Model Decision Tree
# ======================
model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# ======================
# Evaluasi Model
# ======================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ======================
# Feature Importance
# ======================
st.subheader("📊 Top 10 Feature Importance")

importances = pd.Series(
    model.feature_importances_,
    index=X.columns,
    name='importance'
)

importances.index.name = 'Fitur'

importances = (
    importances
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(8,6))

sns.barplot(
    x=importances.values,
    y=importances.index,
    color='steelblue',
    ax=ax
)

ax.set_xlabel("Tingkat Kepentingan (Importance)")
ax.set_title("Top 10 Feature Importance - Decision Tree")

st.pyplot(fig)

# ======================
# Visualisasi Decision Tree
# ======================
st.subheader("🌳 Visualisasi Decision Tree")

fig, ax = plt.subplots(figsize=(36,16))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Tidak Stroke', 'Stroke'],
    filled=True,
    rounded=True,
    fontsize=6,
    ax=ax
)

plt.tight_layout()

st.pyplot(fig)

# ======================
# Confusion Matrix
# ======================
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Tidak Stroke', 'Stroke'],
    yticklabels=['Tidak Stroke', 'Stroke'],
    ax=ax
)

ax.set_xlabel('Prediksi')
ax.set_ylabel('Aktual')
ax.set_title('Confusion Matrix - Decision Tree')

st.pyplot(fig)

# ======================
# Performa Model
# ======================
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

# ======================
# Informasi Tambahan
# ======================
st.info(f"""
Model Decision Tree memiliki:

- Accuracy : {accuracy:.2%}
- Precision : {precision:.2%}
- Recall : {recall:.2%}
- F1-Score : {f1:.2%}

Dataset stroke bersifat tidak seimbang (imbalanced),
karena jumlah pasien stroke jauh lebih sedikit dibandingkan pasien tidak stroke.
""")

# Classification Report
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=['Tidak Stroke', 'Stroke'],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)