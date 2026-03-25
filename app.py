import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from deepface import DeepFace
import os

# ===== TẮT WARNING =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🤖",
    layout="centered"
)

# ===== STYLE (UI đẹp) =====
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1 {
    text-align: center;
    color: #22c55e;
}
.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI Vision App</h1>", unsafe_allow_html=True)
st.write("📷 Nhận diện khuôn mặt, tuổi và vật thể")

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ===== CHỌN NGUỒN ẢNH =====
option = st.radio(
    "Chọn nguồn ảnh:",
    ["📷 Upload ảnh", "📸 Chụp bằng camera"]
)

image = None

# Upload
if option == "📷 Upload ảnh":
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# Camera
elif option == "📸 Chụp bằng camera":
    camera_img = st.camera_input("Chụp ảnh")
    if camera_img:
        image = Image.open(camera_img)

# ===== XỬ LÝ =====
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

        # ===== DEEPFACE (tuổi) =====
        try:
            face = DeepFace.analyze(
                img,
                actions=['age'],
                enforce_detection=False
            )
            age = face[0]['age']
            st.success(f"👤 Tuổi dự đoán: {age}")
        except:
            st.warning("Không phát hiện khuôn mặt")

    # ===== HIỂN THỊ KẾT QUẢ =====
    st.markdown(f"### 👥 Số người phát hiện: **{person_count}**")

    st.markdown("### 📦 Vật thể phát hiện:")
    if count:
        for obj, num in count.items():
            st.write(f"- {obj}: {num}")
    else:
        st.write("Không phát hiện vật thể")

    # ===== VẼ KẾT QUẢ =====
    img_show = results[0].plot()

    st.image(img_show, caption="Kết quả AI", use_container_width=True)
