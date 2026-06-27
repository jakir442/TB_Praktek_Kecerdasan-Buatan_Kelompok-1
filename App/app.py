"""
AI Resume Analyzer Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import pdfplumber
from docx import Document

import re

def parse_resume(text):
    sections = {
        "summary": "",
        "experience": "",
        "education": "",
        "skills": "",
        "other": ""
    }

    lines = text.split("\n")
    current = "other"

    for line in lines:
        l = line.lower()

        if "experience" in l:
            current = "experience"
        elif "education" in l:
            current = "education"
        elif "skill" in l:
            current = "skills"
        elif "summary" in l or "about" in l:
            current = "summary"

        sections[current] += line + "\n"

    return sections

# ============================================================
# UI - AI RESUME ANALYZER (PRO VERSION)
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">

<style>

.main {
    background-color: #0f172a;
}

.hero {
    text-align:center;
    padding:40px;
    border-radius:20px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color:white;
    margin-bottom:20px;
}

.hero h1{
    font-size:48px;
    font-weight:800;
}

.hero p{
    font-size:18px;
    opacity:.9;
}

.metric-card{
    background:#1e293b;
    padding:20px;
    border-radius:16px;
    border:1px solid #334155;
    text-align:center;
    transition:.3s;
}

.metric-card:hover{
    transform:translateY(-5px);
}

.analysis-card{
    background:#111827;
    padding:20px;
    border-radius:16px;
    border:1px solid #374151;
}

.result-card{
    background:linear-gradient(
        135deg,
        #065f46,
        #047857
    );
    border-radius:16px;
    padding:25px;
    text-align:center;
}

.upload-card{
    background:#111827;
    padding:20px;
    border-radius:16px;
    border:2px dashed #475569;
}

.small-title{
    font-size:14px;
    color:#94a3b8;
}

.big-value{
    font-size:30px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# Upload file
def extract_text(file):
    file_type = file.name.split(".")[-1].lower()

    # ---------------- PDF ----------------
    if file_type == "pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    # ---------------- DOCX ----------------
    elif file_type == "docx":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    # ---------------- TXT ----------------
    elif file_type == "txt":
        return str(file.read(), "utf-8")

    else:
        return ""

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "..", "Model")

    model = joblib.load(os.path.join(model_dir, "resume_model.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    role_encoder = joblib.load(os.path.join(model_dir, "role_encoder.pkl"))
    role_map = joblib.load(os.path.join(model_dir, "role_category_map.pkl"))

    return model, tfidf, role_encoder, role_map

model, tfidf, role_encoder, role_map = load_artifacts()

# Preprosessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- CSS MODERN ----------------
st.markdown("""
<style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .card {
        background: #0f172a;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #1f2937;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .upload-box {
        border: 2px dashed #334155;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        background: #0b1220;
    }

    .result-success {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #052e1b, #064e3b);
        border: 1px solid #22c55e;
        text-align: center;
    }

    .result-fail {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2a0f10, #450a0a);
        border: 1px solid #ef4444;
        text-align: center;
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">

<h1>
<i class="bi bi-cpu-fill"></i>
AI Resume Analyzer
</h1>

<p>
Analisis dan Klasifikasi Resume Menggunakan NLP, TF-IDF, SVM dan Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

k1,k2,k3,k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="metric-card">
        <div class="small-title">Model</div>
        <div class="big-value" style="color: white;">TF-IDF</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="metric-card">
        <div class="small-title">Algorithm</div>
        <div class="big-value" style="color: white;">ML</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="metric-card">
        <div class="small-title">Input</div>
        <div class="big-value" style="color: white;">PDF/DOCX</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="metric-card">
        <div class="small-title">Status</div>
        <div class="big-value" style="color: white;">Ready</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("# ⚙ Control Panel")

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf","docx","txt"]
    )

    st.divider()

    st.success("Model Berhasil Dimuat")
# ---------------- MAIN AREA ----------------
col1, col2 = st.columns([1, 1])

with col1:
        st.markdown("## 📄 Resume Preview")

        if uploaded_file is not None:
            resume_text = extract_text(uploaded_file)
            sections = parse_resume(resume_text)

            if sections["summary"].strip():
                st.markdown("#### 🧠 Summary")
                st.info(sections["summary"][:800])

            if sections["experience"].strip():
                st.markdown("#### 💼 Experience")
                st.success(sections["experience"][:1000])

            if sections["education"].strip():
                st.markdown("#### 🎓 Education")
                st.info(sections["education"][:800])

            if sections["skills"].strip():
                st.markdown("#### 🛠 Skills")
                st.warning(sections["skills"][:600])

            with st.expander("📄 Isi Resume Lengkap"):
                st.text_area("", resume_text, height=300)

        else:
            st.info("📂 Unggah resume untuk melihat hasil analisis")

with col2:
    st.markdown("## 🚀 Analysis Panel")

    st.markdown("""
    <div class="card">
        <h4 style="color: white;">AI Model Status</h4>
        <p style="color: white;">Siap menganalisis profil kandidat</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    predict_btn = st.button(
        "🚀 Analyze Resume",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("## 📊 Result Analysis")

    if predict_btn and uploaded_file is not None:
        clean_text = preprocess_text(resume_text)
        text_vector = tfidf.transform([clean_text])
        prediction = model.predict(text_vector)[0]
        predicted_role = role_encoder.inverse_transform([prediction])[0]
        predicted_category = role_map[
            role_map['job_role'] == predicted_role
        ]['category'].values

        predicted_category = predicted_category[0] if len(predicted_category) > 0 else "Unknown"

        st.markdown(f"""
        <div class="result-card">

        <h3>Predicted Category</h3>

        <h1 style="color: white;">{predicted_category}</h1>

        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(text_vector)[0]

            # ==========================
            # TOP 3 PREDICTIONS
            # ==========================
            top_idx = np.argsort(proba)[::-1][:3]

            # ==========================
            # SUMMARY DASHBOARD
            # ==========================
            st.markdown("## 📊 Analysis Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="🎯 Predicted Role",
                    value=predicted_role
                )

            with col2:
                st.metric(
                    label="📁 Category",
                    value=predicted_category
                )

            st.divider()

            # ==========================
            # TOP MATCHING ROLES
            # ==========================
            st.markdown("### 🏆 Posisi yang Paling Sesuai")

            for rank, idx in enumerate(top_idx, start=1):

                try:
                    class_id = model.classes_[idx]
                    role = role_encoder.inverse_transform([class_id])[0]
                except:
                    role = role_encoder.inverse_transform([idx])[0]

                score = proba[idx]
                relative_score = (score / proba[top_idx[0]]) * 100

                with st.container(border=True):

                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"#### #{rank} {role}")

                    with col2:
                        st.metric(
                            label="Kecocokan",
                            value=f"{relative_score:.0f}%"
                        )

                    st.progress(relative_score / 100)

            st.divider()

            # ==========================
            # INSIGHT SECTION
            # ==========================
            st.markdown("## 🤖 AI Insight")

            best_role = predicted_role

            st.info(
                f"""
                Resume yang diunggah memiliki tingkat kecocokan tertinggi dengan posisi
                **{best_role}** pada kategori **{predicted_category}** berdasarkan
                hasil ekstraksi teks menggunakan NLP, pembobotan TF-IDF,
                dan klasifikasi Machine Learning.
                """
            )

            st.success(
                "Resume analysis completed successfully."
            )