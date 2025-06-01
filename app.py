import streamlit as st
import pandas as pd
import joblib
import re
import os
import gdown
import matplotlib.pyplot as plt

# === Load model ===
@st.cache_resource
def load_model():
    os.makedirs("model", exist_ok=True)
    url = "https://drive.google.com/uc?id=19KwWU7HN7JDTXfgbnntReeklWUOSUYsl"  # random_forest_model.pkl
    output = "model/random_forest_model.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return joblib.load(output)

# === Load vectorizer ===
@st.cache_resource
def load_vectorizer():
    url = "https://drive.google.com/uc?id=1UfJ8_vTkNQREQMGmr86Rq5-0eaoI82IS"  # tfidf_vectorizer.pkl
    output = "model/tfidf_vectorizer.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return joblib.load(output)

# === Text cleaning ===
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

# === Single prediction ===
def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

# === Bulk prediction ===
def predict_dataframe(df, model, vectorizer):
    df["cleaned"] = df["komentar"].astype(str).apply(clean_text)
    vect = vectorizer.transform(df["cleaned"])
    df["sentimen"] = model.predict(vect)
    return df

# === Streamlit App ===
st.set_page_config(page_title="🔍 Sentimen Search", page_icon="💬")
st.markdown("<h1 style='text-align: center;'>🔍 Sentimen Search Engine</h1>", unsafe_allow_html=True)

model = load_model()
vectorizer = load_vectorizer()

st.subheader("📥 Masukkan Komentar")
user_input = st.text_input("Contoh: 'Aplikasi ini keren banget!'")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Komentarnya kosong, isi dulu ya 😅")
    else:
        hasil = predict_sentiment(user_input, model, vectorizer)
        st.success(f"Sentimen terdeteksi: **{hasil.upper()}** 🎯")

st.markdown("---")

st.subheader("📁 Upload File Komentar (.csv)")
uploaded_file = st.file_uploader("Upload file dengan kolom 'komentar'", type=["csv"])

if uploaded_file is not None:
    try:
        # Coba baca dengan utf-8 dulu
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            # Kalau gagal, fallback ke ISO-8859-1
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')

        if "komentar" not in df.columns:
            st.error("File harus punya kolom bernama 'komentar'.")
        else:
            st.info("Proses prediksi dimulai...")
            df_hasil = predict_dataframe(df, model, vectorizer)

            st.subheader("📊 Frekuensi Sentimen")
            freq = df_hasil["sentimen"].value_counts()
            st.bar_chart(freq)

            st.subheader("📄 Hasil Pelabelan")
            st.dataframe(df_hasil[["komentar", "sentimen"]])

            csv = df_hasil[["komentar", "sentimen"]].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Hasil ke CSV", data=csv, file_name="hasil_sentimen.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")
