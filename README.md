# 🤖 AI Resume Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/Machine%20Learning-SVM-green.svg">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-orange.svg">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red.svg">
</p>

<p align="center">
  <b>AI Resume Analyzer</b> adalah sistem berbasis Machine Learning dan Natural Language Processing (NLP) yang mampu menganalisis isi resume (CV) dan memprediksi kategori pekerjaan yang paling sesuai berdasarkan pengalaman, keterampilan, pendidikan, dan informasi profesional lainnya.
</p>

---

## 📌 Project Overview

Proses rekrutmen sering kali memerlukan waktu yang cukup lama untuk meninjau dan mengklasifikasikan ribuan resume kandidat.

Proyek ini dikembangkan untuk membantu proses screening awal dengan memanfaatkan teknik **Natural Language Processing (NLP)** dan **Machine Learning**, sehingga sistem dapat memberikan rekomendasi kategori pekerjaan secara otomatis berdasarkan isi resume yang diunggah pengguna.

---

## 🎯 Objectives

* Melakukan analisis teks resume menggunakan NLP.
* Mengubah data teks menjadi representasi numerik menggunakan TF-IDF.
* Mengklasifikasikan resume ke dalam kategori pekerjaan yang sesuai.
* Menyediakan antarmuka berbasis web menggunakan Streamlit.
* Membantu proses screening kandidat secara lebih cepat dan efisien.

---

## 🧠 Technologies Used

### Programming Language

* Python

### Machine Learning

* Support Vector Machine (SVM)

### Natural Language Processing

* TF-IDF Vectorization
* Text Preprocessing

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit

---

## 📂 Dataset

Dataset yang digunakan merupakan kumpulan resume dari berbagai kategori pekerjaan yang digunakan untuk melatih model klasifikasi.

### Dataset Source (Kaggle)

🔗 **Kaggle Dataset:**

https://www.kaggle.com/datasets/trendcart/resume-dataset?resource=download

---

## 🔄 Project Workflow

```text
Resume Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Support Vector Machine (SVM)
       │
       ▼
Model Evaluation
       │
       ▼
Streamlit Web Application
```

## ⚙️ Feature Engineering

Beberapa fitur tambahan yang dibangun pada tahap Feature Engineering:

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| skill_count        | Jumlah skill yang dimiliki kandidat       |
| word_count         | Jumlah kata dalam resume                  |
| char_count         | Jumlah karakter dalam resume              |
| is_experienced     | Status kandidat berpengalaman atau tidak  |
| skill_density      | Kepadatan skill dalam resume              |
| education_level    | Penyederhanaan tingkat pendidikan         |
| job_category_group | Pengelompokan kategori pekerjaan          |
| combined_text      | Gabungan seluruh informasi teks untuk NLP |

---

## 📊 Exploratory Data Analysis (EDA)

Analisis data dilakukan untuk memahami karakteristik dataset, meliputi:

* Distribusi kategori pekerjaan
* Top kategori resume
* Distribusi panjang resume
* Distribusi pengalaman kerja
* Top kata yang paling sering muncul
* Analisis kualitas data

---

## 🤖 Machine Learning Model

### Model Used

Support Vector Machine (SVM)

### Text Representation

TF-IDF Vectorizer

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## 📈 Model Performance

Contoh hasil evaluasi model:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |
| ROC-AUC   | 100%  |

> Nilai di atas dapat berubah sesuai hasil pelatihan model terbaru.

---

## 🖥️ Application Features

### Resume Upload

* Upload file PDF atau DOCX

### Resume Analysis

* Ekstraksi teks otomatis
* Analisis resume menggunakan NLP

### Job Prediction

* Prediksi kategori pekerjaan yang paling sesuai

### Confidence Score

* Menampilkan tingkat keyakinan model terhadap hasil prediksi

### Top Matching Roles

* Menampilkan beberapa kategori pekerjaan dengan skor kecocokan tertinggi

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/USERNAME/AI-Resume-Analyzer.git
```

### Masuk ke Folder Project

```bash
cd AI-Resume-Analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## 📸 Application Preview

Tambahkan screenshot aplikasi di sini.

```text
assets/
├── home.png
├── prediction.png
├── dashboard.png
```

---

## 📁 Project Structure

```text
AI-Resume-Analyzer/
│
├── Dataset/
├── Model/
├── Notebook/
├── Streamlit_App/
│
├── app.py
├── requirements.txt
├── README.md
│
└── assets/
```

---

## 👨‍💻 Team Members

| Name                  |     NIM    |
| --------------------- | ---------- |
| Jakir Apriyan         | 2406004    |
| Rizky Taufik Hidayat  | 2406014    |
| Sandi Febriansah      | 2406001    |
| Maulana Muhammad Zaki | 2406028    |

---

## 🏫 Institution

Institut Teknologi Garut

Program Studi Teknik Informatika

---

## 📄 License

This project is developed for educational and academic purposes.

---

<p align="center">
Made with ❤️ using Python, NLP, TF-IDF, SVM, and Streamlit
</p>
