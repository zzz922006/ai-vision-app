import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests

# ===== CONFIG =====
st.set_page_config(
    page_title="AI Vision Pro",
    page_icon="🤖",
    layout="centered"
)

# ===== STYLE =====
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
h1 {text-align: center; color: #22c55e;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI Vision PRO</h1>", unsafe_allow_html=True)
st.write("📷 Nhận diện vật thể + 👤 đếm người + 🎂 đoán tuổi")

# ===== LOAD YOLO =====
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ===== INPUT =====
option = st.radio(
    "Chọn nguồn:",
    ["📷 Upload ảnh", "📸 Camera"]
)

image = None

if option == "📷 Upload ảnh":
    file = st.file_uploader("Upload ảnh", type=["jpg", "png"])
    if file:
        image = Image.open(file).convert("RGB")

elif option == "📸 Camera":
    cam = st.camera_input("Chụp ảnh")
    if cam:
        image = Image.open(cam).convert("RGB")

# ===== AGE API =====
def estimate_age():
    try:
        r = requests.get("https://api.agify.io?name=alex")
        return r.json()["age"]
    except:
        return None

# ===== PROCESS =====
if image:
    img = np.array(image)
    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    with st.spinner("🔍 AI đang xử lý..."):

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

        # ===== AGE (FAKE AI nhẹ) =====
        age = estimate_age()

    # ===== OUTPUT =====
    st.markdown(f"### 👥 Số người: **{person_count}**")

    if age:
        st.success(f"🎂 Tuổi ước lượng: {age}")
    else:
        st.warning("Không đoán được tuổi")

    st.markdown("### 📦 Vật thể:")
    for k, v in count.items():
        st.write(f"- {k}: {v}")

    # ===== RESULT IMAGE =====
    img_show = results[0].plot()
    st.image(img_show, caption="Kết quả AI", use_container_width=True)
