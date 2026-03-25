import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from deepface import DeepFace
import os

# Tắt warning TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="AI Vision App", layout="centered")

# ===== UI =====
st.markdown("""
    <h1 style='text-align: center; color: #22c55e;'>🤖 AI Vision App</h1>
    <p style='text-align: center;'>Nhận diện khuôn mặt, tuổi và vật thể</p>
""", unsafe_allow_html=True)

# ===== Load model =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ===== Chọn input =====
option = st.radio("Chọn nguồn ảnh:", ["📷 Upload ảnh", "📸 Chụp bằng camera"])

image = None

if option == "📷 Upload ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "📸 Chụp bằng camera":
    camera_img = st.camera_input("Chụp ảnh")
    if camera_img:
        image = Image.open(camera_img)

# ===== Xử lý =====
if image:
    img = np.array(image)

    st.image(image, caption="Ảnh đầu vào", use_column_width=True)

    with st.spinner("🔍 Đang phân tích AI..."):

        # YOLO detect
        results = model(img)

        count = {}
        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                count[name] = count.get(name, 0) + 1

                if name == "person":
                    person_count += 1

        # DeepFace
        try:
            face = DeepFace.analyze(img, actions=['age'], enforce_detection=False)
            age = face[0]['age']
            st.success(f"👤 Tuổi dự đoán: {age}")
        except:
            st.warning("Không phát hiện khuôn mặt")

    # Hiển thị số người
    st.markdown(f"### 👥 Số người phát hiện: **{person_count}**")

    # Hiển thị object
    st.markdown("### 📦 Vật thể phát hiện:")
    for obj, num in count.items():
        st.write(f"- {obj}: {num}")

    # Vẽ bounding box
    img_show = results[0].plot()

    st.image(img_show, caption="Kết quả AI", use_column_width=True)