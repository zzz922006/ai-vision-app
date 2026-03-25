import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Cấu hình giao diện
st.set_page_config(page_title="AI Vision Pro", layout="centered")

# Load model (Sử dụng cache để không load lại mỗi lần nhấn nút)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🤖 AI Vision PRO")

# Chọn nguồn ảnh
option = st.radio("Chọn nguồn:", ["📷 Upload ảnh", "📸 Camera"])

file = None
if option == "📷 Upload ảnh":
    file = st.file_uploader("Chọn ảnh", type=["jpg", "png", "jpeg"])
else:
    file = st.camera_input("Chụp ảnh")

if file:
    # 1. Đọc ảnh từ Streamlit (định dạng PIL RGB)
    image = Image.open(file).convert("RGB")
    
    # 2. Chuyển sang Numpy để YOLO xử lý
    img_array = np.array(image)
    
    # 3. Dự đoán với YOLO
    results = model(img_array)
    
    # 4. Vẽ kết quả (YOLO trả về BGR)
    res_plotted = results[0].plot()
    
    # 5. Hiển thị lên Streamlit (Quan trọng: channels="BGR")
    st.image(res_plotted, channels="BGR", caption="Kết quả từ AI", use_container_width=True)
    
    # Hiển thị thống kê
    st.subheader("📦 Chi tiết vật thể:")
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        st.write(f"- Tìm thấy: **{label}**")
