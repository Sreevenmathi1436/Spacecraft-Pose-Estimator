import streamlit as st
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
import base64
import zipfile
import os

# ✅ Set Streamlit Page Config
st.set_page_config(page_title="Spacecraft Pose Estimation", layout="wide")

# ✅ Register Custom Loss Function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# ✅ Load Pretrained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("satellite_pose_model.h5", custom_objects={"mse": mse})

model = load_model()

# ✅ Function to Preprocess Image
def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Function to Estimate Confidence Score
def compute_confidence(pred_quat, pred_trans):
    quat_norm = np.linalg.norm(pred_quat)
    trans_norm = np.linalg.norm(pred_trans)
    confidence = 100 - (quat_norm + trans_norm)
    return np.clip(confidence, 0, 100)

# ✅ Function to Generate ZIP Report (image + text only, no screen display)
def generate_report(image, image_name, quaternion, translation, confidence):
    # Save image to buffer
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Uploaded Spacecraft Image", fontsize=16)
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)

    # Report content
    report_text = f"""
    🛰️ Spacecraft Pose Estimation Report

    Image Name: {image_name}

    Quaternion Prediction:
    q_x = {quaternion[0]:.4f}
    q_y = {quaternion[1]:.4f}
    q_z = {quaternion[2]:.4f}
    q_w = {quaternion[3]:.4f}

    Translation Prediction:
    x = {translation[0]:.4f}
    y = {translation[1]:.4f}
    z = {translation[2]:.4f}

    Confidence Score: {confidence:.2f}%
    """
    report_buf = io.BytesIO()
    report_buf.write(report_text.encode('utf-8'))
    report_buf.seek(0)

    # ZIP file
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zipf:
        zipf.writestr("spacecraft_image.png", img_buf.getvalue())
        zipf.writestr("pose_report.txt", report_buf.getvalue())
    zip_buf.seek(0)

    # Download link
    b64_zip = base64.b64encode(zip_buf.read()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"pose_report_{timestamp}.zip"
    href = f'<a href="data:application/zip;base64,{b64_zip}" download="{zip_filename}">📦 Download ZIP Report</a>'

    return href

# ✅ Custom CSS
st.markdown("""
<style>
body { background-color: #121212; color: white; }
.title { text-align: center; font-size: 40px; color: #00c3ff; }
.upload-box { border: 2px dashed #00c3ff; padding: 20px; text-align: center; }
.result-box { border-radius: 10px; padding: 20px; background: rgba(0, 0, 0, 0.7); box-shadow: 0 0 20px rgba(255, 255, 255, 0.2); }
</style>
""", unsafe_allow_html=True)

# ✅ Tabs
tabs = st.tabs(["Pose Estimation", "Visualizations"])

with tabs[0]:
    st.markdown("<h1 class='title'>🚀 Spacecraft Pose Estimation</h1>", unsafe_allow_html=True)
    st.write("Upload an image of a spacecraft to estimate its **POSE**.")

    uploaded_file = st.file_uploader("Upload a spacecraft image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_name = os.path.basename(uploaded_file.name)
        st.image(image, caption=f"🛰 Uploaded Image: {image_name}", width=200, use_container_width=False)

        with st.spinner("⏳ Processing... Please wait!"):
            time.sleep(2)
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)

            if isinstance(predictions, list) and len(predictions) == 2:
                quaternion_pred, translation_pred = predictions
                st.session_state["quaternion_pred"] = quaternion_pred
                st.session_state["translation_pred"] = translation_pred
            else:
                st.error("⚠️ Model output format unexpected!")
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='color:#00c3ff;'>🔢 Quaternion (Orientation)</h3>", unsafe_allow_html=True)
            for i, label in enumerate(["q_x", "q_y", "q_z", "q_w"]):
                st.write(f"**{label}:** {quaternion_pred[0][i]:.4f}")

        with col2:
            st.markdown("<h3 style='color:#00c3ff;'>📍 Translation (Position)</h3>", unsafe_allow_html=True)
            for i, label in enumerate(["x", "y", "z"]):
                st.write(f"**{label}:** {translation_pred[0][i]:.4f}")

        confidence = compute_confidence(quaternion_pred[0], translation_pred[0])
        st.metric("📈 Confidence Score", f"{confidence:.2f}%")

        # ✅ Download report only, no display
        href = generate_report(image, image_name, quaternion_pred[0], translation_pred[0], confidence)
        st.markdown(href, unsafe_allow_html=True)

        st.success("✅ Pose estimation complete!")

with tabs[1]:
    st.header("📊 Visualizations & Analysis")

    if "quaternion_pred" in st.session_state and "translation_pred" in st.session_state:
        quaternion_values = st.session_state["quaternion_pred"][0]
        translation_values = st.session_state["translation_pred"][0]

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(x=["q_x", "q_y", "q_z", "q_w"], y=quaternion_values, color=["q_x", "q_y", "q_z", "q_w"], title="Quaternion Values")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(x=["x", "y", "z"], y=translation_values, color=["x", "y", "z"], title="Translation Values")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🕵️‍♂️ Radar Chart: Quaternion Overview")
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(r=quaternion_values, theta=["q_x", "q_y", "q_z", "q_w"], fill='toself'))
        radar_fig.update_layout(title="Radar Chart of Quaternion Predictions", showlegend=False, width=500, height=400)
        st.plotly_chart(radar_fig)

        st.subheader("📌 3D Scatter Plot: Translation Position")
        scatter_fig = px.scatter_3d(x=[translation_values[0]], y=[translation_values[1]], z=[translation_values[2]],
                                    labels={"x": "X Position", "y": "Y Position", "z": "Z Position"},
                                    title="Spacecraft Position in 3D Space")
        scatter_fig.update_traces(marker=dict(size=8, color="#ff9900"))
        st.plotly_chart(scatter_fig)

# ✅ Footer
st.markdown("---")
