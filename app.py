import streamlit as st
import tempfile
import os

# Import โมดูลที่เราสร้างขึ้นเอง
from model import YoloSegmentationModel
from inference import run_inference
from utils import load_image_from_upload, draw_segmentation_results

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Segmentation App", layout="wide")
st.title("🧩 Modular AI Segmentation App")

# --- 1. Load Model Section ---
# ใช้ @st.cache_resource เพื่อโหลดโมเดลแค่ครั้งเดียว ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def get_loaded_model(pt_file_path):
    model_instance = YoloSegmentationModel()
    success = model_instance.load_weights(pt_file_path)
    if success:
        return model_instance
    else:
        return None

# Sidebar สำหรับ Config
st.sidebar.header("⚙️ Configuration")
model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])
conf_score = st.sidebar.slider("Confidence Score", 0.0, 1.0, 0.25)

# --- 2. Main Logic ---
if model_file:
    # Save temp file เพื่อให้ YOLO library อ่าน path ได้
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(model_file.getvalue())
        temp_model_path = tmp.name

    # เรียกใช้ Class จาก model.py
    model_wrapper = get_loaded_model(temp_model_path)
    
    if model_wrapper:
        st.sidebar.success("✅ Model Loaded!")
    else:
        st.sidebar.error("❌ Failed to load model.")
        st.stop()

    # --- 3. Image Input & Processing ---
    uploaded_image = st.file_uploader("Upload Image to Analyze", type=['jpg', 'png', 'jpeg'])

    if uploaded_image:
        col1, col2 = st.columns(2)
        
        # 3.1 ใช้ function จาก utils.py แปลงภาพ
        original_img = load_image_from_upload(uploaded_image)
        
        with col1:
            st.info("Original Image")
            st.image(original_img, use_container_width=True)

        # ปุ่มกด Predict
        if st.button("🔍 Run Inference"):
            with st.spinner("Processing..."):
                try:
                    # 3.2 ใช้ function จาก inference.py เพื่อทำนาย
                    raw_results = run_inference(model_wrapper, original_img, conf_score)
                    
                    # 3.3 ใช้ function จาก utils.py เพื่อวาดภาพผลลัพธ์
                    result_img, found = draw_segmentation_results(original_img, raw_results)
                    
                    with col2:
                        if found:
                            st.success(f"Segmentation Complete!")
                        else:
                            st.warning("No objects found.")
                        
                        st.image(result_img, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error during inference: {e}")

else:
    st.info("👈 Please upload a .pt model file in the sidebar to start.")
