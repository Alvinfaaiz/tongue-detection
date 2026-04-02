import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Tongue Diabetes Detection",
    layout="centered"
)

st.title("TongueScan")
st.write("Deteksi dini diabetes melalui gambar lidah menggunakan model CNN dan MobileNetV2")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    model_cnn = tf.keras.models.load_model("tongue_cnn_model.keras")
    model_mobilenet = tf.keras.models.load_model("tongue_mobilenetv2_binary.keras")
    return model_cnn, model_mobilenet

model_cnn, model_mobilenet = load_models()

# ==========================
# Preprocessing
# ==========================

# Untuk CNN biasa (128x128 + /255)
def preprocess_cnn(image):
    image = image.resize((128,128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Untuk MobileNet (224x224 + preprocess_input)
def preprocess_mobilenet(image):
    image = image.resize((224,224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ==========================
# Upload Image
# ==========================
uploaded_file = st.file_uploader(
    "Upload Gambar Lidah",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # ==========================
    # Model 1 - CNN
    # ==========================
    processed_cnn = preprocess_cnn(image)
    pred_cnn = model_cnn.predict(processed_cnn)[0][0]

    # ==========================
    # Model 2 - MobileNet
    # ==========================
    processed_mobilenet = preprocess_mobilenet(image)
    pred_mobilenet = model_mobilenet.predict(processed_mobilenet)[0][0]

    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    # ===== CNN =====
    with col1:
        st.markdown("### Model CNN")

        if pred_cnn > 0.5:
            confidence = pred_cnn * 100
            st.error("Non-Diabetes")
        else:
            confidence = (1 - pred_cnn) * 100
            st.success("Diabetes")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))

    # ===== MobileNet =====
    with col2:
        st.markdown("### Model MobileNetV2")

        if pred_mobilenet > 0.5:
            confidence = pred_mobilenet * 100
            st.error("Non-Diabetes")
        else:
            confidence = (1 - pred_mobilenet) * 100
            st.success("Diabetes")

        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(float(confidence / 100))
