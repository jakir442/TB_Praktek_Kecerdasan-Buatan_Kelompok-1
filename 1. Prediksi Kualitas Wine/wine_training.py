import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn modules untuk pemrosesan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 5 Algoritma Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrik Evaluasi
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

print("📥 TAHAP 1: PENGUMPULAN DATA")

# Load dataset dari file CSV
df = pd.read_csv("WineQT.csv")
df['quality_category'] = np.where(
    df['quality'] <= 5,
    'Buruk',
    np.where(
        df['quality'] <= 8,
        'Sedang',
        'Bagus'
    )
)

print(f"✓ Dataset berhasil dimuat dari 'WineQT.csv'")
print(f"  Jumlah baris  : {df.shape[0]}")
print(f"  Jumlah kolom  : {df.shape[1]}")
print(f"  Nama Kolom    : {list(df.columns)}")

print("\n📊 5 Baris Pertama:")
print(df.head())

print("🔍 TAHAP 2: EKSPLORASI DATA (EDA)")

fitur_num = [
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
    'alcohol'
]

target_col = 'quality_category'

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("EDA - Dataset WineQT", fontsize=24, fontweight="bold")

# 1. Scatter Plot
sns.scatterplot(
    data=df,
    x="alcohol",
    y="pH",
    hue="quality_category",
    palette="Set2",
    s=80,
    ax=axes[0, 0]
)

axes[0, 0].set_title("Scatter Plot: Alcohol vs pH", fontsize=17)

# 2. Histogram
sns.histplot(
    data=df,
    x="alcohol",
    hue="quality_category",
    kde=True,
    bins=15,
    palette="Set2",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Distribusi Alcohol", fontsize=17)

# 3. Boxplot
df_melt = df.melt(
    id_vars="quality_category",
    value_vars=fitur_num,
    var_name="Fitur",
    value_name="Nilai"
)

sns.boxplot(
    data=df_melt,
    x="Fitur",
    y="Nilai",
    hue="quality_category",
    palette="Set2",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Boxplot Fitur per Quality", fontsize=17)
axes[1, 0].tick_params(axis='x', rotation=35)

# 4. Heatmap Korelasi
corr_df = pd.concat(
    [df[fitur_num], pd.get_dummies(df['quality_category'])],
    axis=1
)

corr_quality = corr_df.corr().loc[
    fitur_num,
    df['quality_category'].unique()
]

sns.heatmap(
    corr_quality,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Korelasi Fitur terhadap Quality",
    fontsize=17
)

plt.tight_layout()
plt.show()

print("⚙️ TAHAP 3: DATA PROCESSING")

# 3a. Cek Missing Values
missing = df.isnull().sum()

print(f"""
[3a] Missing Values:

{missing[missing > 0] if missing.sum() > 0 else 'Tidak ada missing value.'}
""")

# 3b. Cek & Hapus Duplikasi
duplikat = df.duplicated().sum()

print(f"[3b] Baris Duplikat: {duplikat}")

if duplikat > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("✓ Duplikat berhasil dihapus.")

# 3c. Deteksi Outlier (IQR Method)
print("\n[3c] Deteksi Outlier (Metode IQR):")

for col in fitur_num:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_out = len(
        df[
            (df[col] < lower) |
            (df[col] > upper)
        ]
    )

    print(f"{col:<25}: {n_out} outlier")

print("\n✅ Data Processing selesai.")

print("🧠 TAHAP 4: FEATURE ENGINEERING")

# 4a. Membuat Fitur Baru
df['sulfur_ratio'] = (
    df['free sulfur dioxide'] /
    df['total sulfur dioxide']
)

df['acidity_ratio'] = (
    df['fixed acidity'] /
    df['volatile acidity']
)

print("""
✓ Fitur baru berhasil ditambahkan:
- sulfur_ratio
- acidity_ratio
""")

# 4b. Encode Target Variable
le = LabelEncoder()

df['quality_encoded'] = le.fit_transform(df['quality_category'])

print(f"""
✓ Encoding kelas berhasil:

{dict(zip(le.classes_, le.transform(le.classes_)))}
""")

# 4c. Persiapan X dan y
fitur_all = fitur_num + [
    'sulfur_ratio',
    'acidity_ratio'
]

X = df[fitur_all].values
y = df['quality_encoded'].values

# Standardisasi
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(f"""
✓ Fitur final:
{fitur_all}
""")

print(f"""
✓ Mean setelah scaling:
{X_scaled.mean(axis=0).round(2)}
""")

print(f"""
✓ Std setelah scaling:
{X_scaled.std(axis=0).round(2)}
""")

print("🧠 TAHAP 5: MODELLING")

from sklearn.model_selection import cross_val_score

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"""
Ukuran Data:
Train : {X_train.shape}
Test  : {X_test.shape}
""")

# Inisialisasi Model
models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=500,
        random_state=42
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM (RBF)":
    SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "K-Nearest Neighbors":
    KNeighborsClassifier(
        n_neighbors=5
    ),
}

print("\n📊 5-Fold Cross Validation")

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores

    print(f"""
{name}

Mean Accuracy : {scores.mean():.4f}
Std Dev       : {scores.std():.4f}
""")

# Training Semua Model
for name, model in models.items():
    model.fit(X_train, y_train)

# Model Terbaik
best_name = max(
    cv_results,
    key=lambda k: cv_results[k].mean()
)

best_model = models[best_name]

print(f"""
✓ Model terbaik adalah:
{best_name}
""")

print("🎯 TAHAP 6: DATA EVALUATION")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

eval_results = {}

print(f"{'Model':<25} {'Akurasi':>9} {'Presisi':>9} {'Recall':>9} {'F1':>9}")
print("-" * 60)

# Evaluasi Semua Model
for name, model in models.items():

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    eval_results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

    print(f"{name:<25} {acc:>9.4f} {prec:>9.4f} {rec:>9.4f} {f1:>9.4f}")

# Prediksi Model Terbaik
y_pred_best = best_model.predict(X_test)

print(f"\n📋 Classification Report - {best_name}:\n")

print(classification_report(
    y_test,
    y_pred_best,
    target_names=[str(x) for x in le.classes_]
))

# =========================
# TABEL PERBANDINGAN
# =========================

print("\n" + "=" * 65)
print("📊 TABEL PERBANDINGAN AKURASI MODEL")
print("=" * 65)

for name, result in eval_results.items():

    print(f"""
Model       : {name}
Accuracy    : {result['Accuracy']:.4f}
Precision   : {result['Precision']:.4f}
Recall      : {result['Recall']:.4f}
F1-Score    : {result['F1']:.4f}
""")

print("=" * 65)

print(f"""
Interpretasi:

Model terbaik adalah {best_name}
dengan akurasi sebesar
{eval_results[best_name]['Accuracy']:.4f}

Model ini memiliki performa terbaik
dibanding model lainnya
dalam melakukan prediksi kualitas wine.
""")

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred_best
)

print("\n" + "=" * 65)
print("📌 INTERPRETASI CONFUSION MATRIX")
print("=" * 65)

print("""
Confusion Matrix digunakan
untuk melihat jumlah prediksi benar
dan salah pada setiap kelas quality wine.
""")

print("""
Semakin besar nilai diagonal utama,
maka model semakin baik
dalam melakukan prediksi.
""")

print("""
Berdasarkan confusion matrix:

1. Model paling baik mengenali quality 5 dan 6.

2. Kesalahan prediksi terbesar
terjadi antara quality 5 dan 6.

3. Quality 3, 4, dan 8
sulit diprediksi karena
jumlah data sangat sedikit.

4. Dataset mengalami
imbalanced dataset.
""")

print("\n📌 HASIL CONFUSION MATRIX:\n")
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n" + "=" * 65)
print("📌 ANALISIS CLASSIFICATION REPORT")
print("=" * 65)

print("""
1. Precision
Menunjukkan ketepatan model
saat memprediksi suatu kelas.
""")

print("""
2. Recall
Menunjukkan kemampuan model
menemukan data aktual.
""")

print("""
3. F1-Score
Merupakan keseimbangan antara
precision dan recall.
""")

print("""
4. Support
Menunjukkan jumlah data
pada tiap kelas.
""")

print(f"""
Berdasarkan classification report,
model {best_name}
cukup baik mengenali
quality 5 dan 6,
namun masih kesulitan
mengenali quality 3, 4, dan 8.
""")

# =========================
# VISUALISASI
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

fig.suptitle(
    f"Evaluasi Model - {best_name}",
    fontsize=20,
    fontweight="bold"
)

# Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[0]
)

axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Prediksi")
axes[0].set_ylabel("Aktual")

# Barplot Akurasi
names = list(eval_results.keys())

accurs = [
    eval_results[n]["Accuracy"]
    for n in names
]

colors = [
    '#e74c3c'
    if n == best_name
    else '#3498db'
    for n in names
]

bars = axes[1].barh(
    names,
    accurs,
    color=colors
)

axes[1].set_xlim(0.50, 1.01)

axes[1].set_xlabel("Akurasi")

axes[1].set_title(
    "Perbandingan Akurasi Model"
)

for bar, val in zip(bars, accurs):

    axes[1].text(
        val + 0.002,
        bar.get_y() + bar.get_height()/2,
        f"{val:.4f}",
        va="center"
    )

plt.tight_layout()
plt.show()

print("🚀 TAHAP 7: DEPLOYMENT SEDERHANA")

import joblib

# Simpan model
joblib.dump(best_model, "wine_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("""
✓ Model berhasil disimpan:
- wine_model.pkl
- scaler.pkl
- label_encoder.pkl
""")

# Fungsi Prediksi
def predict_quality(
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
    alcohol
):

    df_input = pd.DataFrame(
        [[
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
            alcohol
        ]],
        columns=fitur_num
    )

    # Feature Engineering otomatis
    df_input['sulfur_ratio'] = (
        df_input['free sulfur dioxide'] /
        df_input['total sulfur dioxide']
    )

    df_input['acidity_ratio'] = (
        df_input['fixed acidity'] /
        df_input['volatile acidity']
    )

    # Scaling
    input_scaled = scaler.transform(
        df_input[fitur_all].values
    )

    # Predict
    pred = best_model.predict(input_scaled)[0]

    proba = best_model.predict_proba(
        input_scaled
    )[0]

    return {

        "prediksi":
        le.inverse_transform([pred])[0],

        "probabilitas": {

            str(le.inverse_transform([i])[0]):
            round(float(proba[i]), 4)

            for i in range(len(le.classes_))
        }
    }

print("\n🎯 Contoh Prediksi Data Baru:\n")

contoh_data = [

    (
        7.4, 0.70, 0.00,
        1.9, 0.076,
        11.0, 34.0,
        0.9978, 3.51,
        0.56, 9.4
    ),

    (
        8.5, 0.28, 0.56,
        1.8, 0.092,
        35.0, 103.0,
        0.9969, 3.30,
        0.75, 10.5
    ),

    (
        11.2, 0.28, 0.56,
        1.9, 0.075,
        17.0, 60.0,
        0.9980, 3.16,
        0.58, 9.8
    ),
]

for data in contoh_data:

    hasil = predict_quality(*data)

    print(f"""
Input Data:
{data}

Prediksi Quality:
{hasil['prediksi']}

Probabilitas:
{hasil['probabilitas']}
""")

print("=" * 65)
print("🏁 PIPELINE MACHINE LEARNING SELESAI!")
print("=" * 65)

print("""
Tahap 1 → Data Collection
Tahap 2 → EDA
Tahap 3 → Data Processing
Tahap 4 → Feature Engineering
Tahap 5 → Modelling
Tahap 6 → Data Evaluation
Tahap 7 → Deployment Sederhana
""")

print(f"""
Model Terbaik : {best_name}
Akurasi Uji   : {eval_results[best_name]['Accuracy']:.4f}
F1-Score      : {eval_results[best_name]['F1']:.4f}
""")