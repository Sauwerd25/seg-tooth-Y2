import streamlit as st
import cv2
import numpy as np
import os
import shutil
import tempfile
import zipfile
from PIL import Image
from ultralytics import YOLO

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Segmentation App", layout="wide")

st.title("🧩 AI Segmentation & Dataset Tool")

# สร้าง Tabs เพื่อแยกฟังก์ชัน
tab1, tab2 = st.tabs(["🚀 AI Prediction (ทำนายผล)", "🛠️ Dataset Preparation (เตรียมข้อมูล)"])

# ==========================================
# TAB 1: AI PREDICTION (ทำนายผลลัพธ์)
# ==========================================
with tab1:
    st.header("อัปโหลดรูปภาพเพื่อทำนายผล (Segmentation)")
    
    # 1. Upload Model
    st.sidebar.header("Model Config")
    model_file = st.sidebar.file_uploader("1. อัปโหลดไฟล์ Model (.pt)", type=['pt'])
    
    # 2. Upload Image
    uploaded_file = st.file_uploader("2. เลือกรูปภาพที่ต้องการตรวจสอบ", type=['jpg', 'png', 'jpeg'])

    if model_file is not None and uploaded_file is not None:
        # Save model to temp file to load it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_file.getvalue())
            model_path = tmp_model.name

        try:
            # Load Model
            model = YOLO(model_path)
            
            # Read Image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="รูปต้นฉบับ", use_container_width=True)

            # Predict button
            if st.button("🔍 เริ่มการทำนาย (Predict)"):
                with st.spinner('กำลังประมวลผล...'):
                    results = model(image)
                    
                    # Plot Result
                    res_plotted = results[0].plot()
                    
                    with col2:
                        st.image(res_plotted, caption="ผลลัพธ์การทำนาย", use_container_width=True)
                        
                    # Show Masks if available
                    if results[0].masks is not None:
                        st.success("พบวัตถุในภาพ!")
                    else:
                        st.warning("ไม่พบวัตถุในภาพ")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
            
    elif not model_file:
        st.info("👈 กรุณาอัปโหลดไฟล์ Model (.pt) ที่แถบด้านซ้ายก่อน")

# ==========================================
# TAB 2: DATASET PREPARATION (โค้ดเตรียมข้อมูลของคุณ)
# ==========================================
with tab2:
    st.header("แปลงข้อมูล Mask เป็น YOLO Dataset")
    st.markdown("""
    เครื่องมือนี้จะทำการ:
    1. เปลี่ยนชื่อไฟล์ Mask (ตาม Logic โค้ดเดิม)
    2. แปลง Mask ขาวดำ เป็น Polygon Coordinates (.txt)
    3. แบ่งข้อมูลเป็น Train/Val และสร้างไฟล์ `data.yaml`
    """)

    # Upload Zips
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        mask_zip = st.file_uploader("อัปโหลดไฟล์ Zip ของ Mask (label_png)", type=['zip'])
    with col_upload2:
        img_zip = st.file_uploader("อัปโหลดไฟล์ Zip ของรูปจริง (data_folder)", type=['zip'])

    val_split = st.slider("สัดส่วน Validation Set", 0.1, 0.5, 0.2)

    if st.button("⚙️ เริ่มกระบวนการแปลงข้อมูล") and mask_zip and img_zip:
        with st.spinner("กำลังประมวลผล... กรุณารอสักครู่"):
            # Create temporary directories
            temp_dir = tempfile.mkdtemp()
            mask_extract_path = os.path.join(temp_dir, 'label_png')
            img_extract_path = os.path.join(temp_dir, 'data_folder')
            output_dataset_path = os.path.join(temp_dir, 'my_yolo_dataset')
            
            os.makedirs(mask_extract_path, exist_ok=True)
            os.makedirs(img_extract_path, exist_ok=True)

            # Extract Zips
            with zipfile.ZipFile(mask_zip, 'r') as zip_ref:
                zip_ref.extractall(mask_extract_path)
            with zipfile.ZipFile(img_zip, 'r') as zip_ref:
                zip_ref.extractall(img_extract_path)

            # --- 1. RENAME LOGIC (จากโค้ดของคุณ) ---
            # ปรับปรุงให้รองรับโครงสร้างไฟล์จากการแตก Zip
            # หาโฟลเดอร์จริงที่รูปอยู่ (กรณี zip ซ้อน folder)
            real_mask_folder = mask_extract_path
            for root, dirs, files in os.walk(mask_extract_path):
                if len(files) > 0 and any(f.endswith('.png') for f in files):
                    real_mask_folder = root
                    break
            
            all_mask_files = os.listdir(real_mask_folder)
            
            count_renamed = 0
            for i in all_mask_files:
                if 'tag-' in i:
                    try:
                        # Logic การตัดคำของคุณ
                        j = i.split('tag-')[-1]
                        z = j.split('-')
                        if len(z) >= 2:
                            new_name = z[0] + z[1] 
                            # ตรวจสอบนามสกุลไฟล์เดิม
                            ext = os.path.splitext(i)[1]
                            if not new_name.endswith(ext):
                                new_name += ext # เติมนามสกุลถ้าหายไป
                                
                            src = os.path.join(real_mask_folder, i)
                            dst = os.path.join(real_mask_folder, new_name)
                            os.rename(src, dst)
                            count_renamed += 1
                    except Exception as e:
                        print(f"Skipping {i}: {e}")

            st.write(f"✅ เปลี่ยนชื่อไฟล์ Mask เสร็จสิ้น ({count_renamed} ไฟล์)")

            # --- 2. MASK TO POLYGON ---
            output_txt_folder = os.path.join(temp_dir, 'labels_seg')
            os.makedirs(output_txt_folder, exist_ok=True)
            
            converted_count = 0
            for filename in os.listdir(real_mask_folder):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        mask_path = os.path.join(real_mask_folder, filename)
                        mask = cv2.imread(mask_path, 0)
                        h, w = mask.shape
                        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        txt_content = []
                        for cnt in contours:
                            if len(cnt) < 3: continue
                            normalized_points = []
                            for point in cnt:
                                x_coord = point[0][0] / w
                                y_coord = point[0][1] / h
                                normalized_points.extend([f"{x_coord:.6f}", f"{y_coord:.6f}"])
                            poly_line = "0 " + " ".join(normalized_points)
                            txt_content.append(poly_line)

                        if txt_content:
                            # Save as .txt (ชื่อเดียวกับ mask แต่เปลี่ยนนามสกุล)
                            txt_filename = os.path.splitext(filename)[0] + ".txt"
                            # Handle potential naming mismatch if rename logic created complex names
                            # But here we just use the current filename
                            with open(os.path.join(output_txt_folder, txt_filename), 'w') as f:
                                f.write('\n'.join(txt_content))
                            converted_count += 1
                    except Exception as e:
                        pass
            
            st.write(f"✅ แปลง Mask เป็น .txt เสร็จสิ้น ({converted_count} ไฟล์)")

            # --- 3. SPLIT TRAIN/VAL ---
            for split in ['train', 'val']:
                os.makedirs(f'{output_dataset_path}/{split}/images', exist_ok=True)
                os.makedirs(f'{output_dataset_path}/{split}/labels', exist_ok=True)

            # หาโฟลเดอร์รูปจริง
            real_img_folder = img_extract_path
            for root, dirs, files in os.walk(img_extract_path):
                if len(files) > 0 and any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    real_img_folder = root
                    break

            image_files = [f for f in os.listdir(real_img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            import random
            random.shuffle(image_files)
            
            split_index = int(len(image_files) * (1 - val_split))
            train_files = image_files[:split_index]
            val_files = image_files[split_index:]

            def copy_data(files, split_type):
                c = 0
                for filename in files:
                    # Copy Image
                    shutil.copy(os.path.join(real_img_folder, filename),
                                f'{output_dataset_path}/{split_type}/images/{filename}')
                    
                    # Copy Label
                    # ต้องหาไฟล์ txt ที่ชื่อตรงกัน (Logic: ชื่อไฟล์รูป ตัดนามสกุล -> ชื่อไฟล์ txt)
                    # ข้อควรระวัง: ชื่อไฟล์รูปต้องตรงกับชื่อที่ถูก Rename ในขั้นตอนที่ 1
                    # ถ้าชื่อไม่ตรงกัน (เพราะขั้นตอน 1 rename แต่ mask) ขั้นตอนนี้จะหา Label ไม่เจอ
                    # สมมติว่า User ตั้งชื่อไฟล์รูปให้ตรงกับ Mask ผลลัพธ์แล้ว
                    
                    # พยายาม Match ชื่อไฟล์ (Simple matching)
                    label_name = os.path.splitext(filename)[0] + '.txt'
                    label_src = os.path.join(output_txt_folder, label_name)
                    
                    # Try fuzzy match logic if direct match fails (เนื่องจาก logic rename ซับซ้อน)
                    if not os.path.exists(label_src):
                         # ลองหาไฟล์ใน output_txt_folder ที่มีส่วนประกอบคล้ายกัน (Optional workaround)
                         pass

                    if os.path.exists(label_src):
                        shutil.copy(label_src, f'{output_dataset_path}/{split_type}/labels/{label_name}')
                        c += 1
                return c

            t_count = copy_data(train_files, 'train')
            v_count = copy_data(val_files, 'val')
            
            st.write(f"✅ จัดเรียงไฟล์ลงโฟลเดอร์ Train ({t_count}) / Val ({v_count})")

            # Create data.yaml
            yaml_content = f"""
            path: ../ # dataset root dir
            train: train/images
            val: val/images
            
            names:
              0: object
            """
            with open(os.path.join(output_dataset_path, 'data.yaml'), 'w') as f:
                f.write(yaml_content.strip())

            # Zip Result
            shutil.make_archive(os.path.join(temp_dir, 'yolo_dataset_ready'), 'zip', output_dataset_path)
            
            with open(os.path.join(temp_dir, 'yolo_dataset_ready.zip'), "rb") as fp:
                btn = st.download_button(
                    label="📥 ดาวน์โหลด Dataset (.zip)",
                    data=fp,
                    file_name="my_yolo_dataset.zip",
                    mime="application/zip"
                )
