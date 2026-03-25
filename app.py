import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from deepface import DeepFace
import os

# ===== FIX WARNING =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ===== CONFIG =====
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🤖",
    layout="centered"
)

# ===== UI =====
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
h1 {
    text-align: center;
    color: #22c55e;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI Vision App</h1>", unsafe_allow_html=True)
st.write("📷 Nhận diện người, vật thể và tuổi")

# ===== LOAD MODEL (QUAN TRỌNG) =====
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# ===== INPUT =====
option = st.radio(
    "Chọn nguồn ảnh:",
    ["📷 Upload ảnh", "📸 Chụp bằng camera"]
)

image = None

if option == "📷 Upload ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "📸 Chụp bằng camera":
    camera_img = st.camera_input("Chụp ảnh")
    if camera_img:
        image = Image.open(camera_img).convert("RGB")

# ===== PROCESS =====
if image:
    img = np.array(image)

    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    with st.spinner("🔍 Đang xử lý AI..."):

        # ===== YOLO =====
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

        # ===== DEEPFACE (GIẢM LOAD) =====
        age = None
        try:
            face = DeepFace.analyze(
                img,
                actions=['age'],
                enforce_detection=False,
                detector_backend='opencv'   # 🔥 nhẹ hơn mặc định
            )
            age = face[0]['age']
        except:
            pass

    # ===== OUTPUT =====
    if age:
        st.success(f"👤 Tuổi dự đoán: {age}")
    else:
        st.warning("Không phát hiện khuôn mặt")

    st.markdown(f"### 👥 Số người: **{person_count}**")

    st.markdown("### 📦 Vật thể phát hiện:")
    if count:
        for obj, num in count.items():
            st.write(f"- {obj}: {num}")
    else:
        st.write("Không phát hiện vật thể")

    # ===== IMAGE RESULT =====
    img_show = results[0].plot()
    st.image(img_show, caption="Kết quả AI", use_container_width=True)
