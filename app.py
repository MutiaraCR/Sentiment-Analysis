import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
import os
import gdown

# ====== LOAD MODEL DARI GOOGLE DRIVE ======
@st.cache_resource
def load_model():
    import pickle  # Biar aman, kita pakai pickle daripada joblib

    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=1S_V-gdK4fOOA0PEiFZDN4HMih5DaI0Ug"
    output = "model/random_forest_model.pkl"
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    with open(output, "rb") as f:
        model = pickle.load(f)
    
    return model

# ====== PREPROCESSING ======
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

# ====== PREDIKSI MANUAL ======
def predict_sentiment(text, model):
    cleaned = clean_text(text)
    return model.predict([cleaned])[0]

# ====== PREDIKSI DARI FILE ======
def predict_dataframe(df, model):
    df["cleaned"] = df["komentar"].astype(str).apply(clean_text)
    df["sentimen"] = model.predict(df["cleaned"])
    return df

# ====== UI START ======
st.set_page_config(page_title="🔍 Sentimen Search", page_icon="💬")
st.markdown("<h1 style='text-align: center;'>🔍 Sentimen Search Engine</h1>", unsafe_allow_html=True)

model = load_model()

# === INPUT MANUAL ===
st.subheader("📥 Masukkan Komentar")
user_input = st.text_input("Contoh: 'Aplikasi ini keren banget!'")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Komentarnya kosong, isi dulu ya 😅")
    else:
        hasil = predict_sentiment(user_input, model)
        st.success(f"Sentimen terdeteksi: **{hasil.upper()}** 🎯")

st.markdown("---")

# === UPLOAD FILE ===
st.subheader("📁 Upload File Komentar (.csv)")
uploaded_file = st.file_uploader("Upload file dengan kolom 'komentar'", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "komentar" not in df.columns:
            st.error("File harus punya kolom bernama 'komentar'.")
        else:
            st.info("Proses prediksi dimulai...")
            df_hasil = predict_dataframe(df, model)

            # Bar chart
            st.subheader("📊 Frekuensi Sentimen")
            freq = df_hasil["sentimen"].value_counts()
            st.bar_chart(freq)

            # Tabel hasil
            st.subheader("📄 Hasil Pelabelan")
            st.dataframe(df_hasil[["komentar", "sentimen"]])

            # Tombol download hasil
            csv = df_hasil[["komentar", "sentimen"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Hasil ke CSV",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
