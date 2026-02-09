import os
import cv2
import numpy as np
import shutil
import random
from ultralytics import YOLO

# ==========================================
# 1. ส่วนตั้งค่า Path (แก้ไขตรงนี้ให้ตรงกับเครื่องคุณ)
# ==========================================
# โฟลเดอร์ที่มีรูป Mask ขาวดำ
MASK_FOLDER_RAW = '/content/label_png' 
# โฟลเดอร์รูปภาพต้นฉบับ
IMAGE_FOLDER_RAW = '/content/data_folder1'
# โฟลเดอร์ปลายทางที่จะสร้างเป็น Dataset พร้อม Train
DATASET_ROOT = '/content/my_yolo_dataset'

# ชื่อ Model เริ่มต้น (ใช้โมเดลขนาดเล็ก nano สำหรับ segmentation)
BASE_MODEL = 'yolo8n-seg.pt' 

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูล (แปลง Mask -> Polygon -> Split Train/Val)
# ==========================================
def prepare_data():
    print("🚀 เริ่มกระบวนการเตรียมข้อมูล...")
    
    # 2.1 เปลี่ยนชื่อไฟล์ Mask (ตาม Logic เดิมของคุณ)
    print("--- ขั้นตอนที่ 1: เปลี่ยนชื่อไฟล์ Mask ---")
    all_mask_files = os.listdir(MASK_FOLDER_RAW)
    n_renamed = 0
    for i in all_mask_files:
        try:
            if 'tag-' in i:
                j = i.split('tag-')[-1]
                z = j.split('-')
                new_name = z[0] + z[1] # Logic การรวมชื่อ
                
                # ตรวจสอบนามสกุลไฟล์
                if not new_name.endswith('.png'):
                    new_name += '.png'
                    
                src = os.path.join(MASK_FOLDER_RAW, i)
                dst = os.path.join(MASK_FOLDER_RAW, new_name)
                os.rename(src, dst)
                n_renamed += 1
        except Exception as e:
            print(f"Skipping {i}: {e}")
    print(f"เปลี่ยนชื่อเสร็จสิ้น {n_renamed} ไฟล์")

    # 2.2 แปลง Mask เป็น .txt (Polygon)
    print("--- ขั้นตอนที่ 2: แปลง Mask เป็น Polygon ---")
    temp_label_dir = 'temp_labels_txt'
    os.makedirs(temp_label_dir, exist_ok=True)
    
    for filename in os.listdir(MASK_FOLDER_RAW):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(MASK_FOLDER_RAW, filename)
            mask = cv2.imread(mask_path, 0)
            if mask is None: continue
            
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
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(temp_label_dir, txt_filename), 'w') as f:
                    f.write('\n'.join(txt_content))

    # 2.3 แบ่ง Train/Val และย้ายไฟล์
    print("--- ขั้นตอนที่ 3: จัดเรียงไฟล์ลง Dataset ---")
    for split in ['train', 'val']:
        os.makedirs(f'{DATASET_ROOT}/{split}/images', exist_ok=True)
        os.makedirs(f'{DATASET_ROOT}/{split}/labels', exist_ok=True)

    image_files = [f for f in os.listdir(IMAGE_FOLDER_RAW) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)
    
    val_split = 0.2
    split_index = int(len(image_files) * (1 - val_split))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def copy_files(files, split_type):
        for filename in files:
            # Copy Image
            src_img = os.path.join(IMAGE_FOLDER_RAW, filename)
            if os.path.exists(src_img):
                shutil.copy(src_img, f'{DATASET_ROOT}/{split_type}/images/{filename}')
            
            # Copy Label (ชื่อไฟล์รูปตัดนามสกุล + .txt)
            label_name = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(temp_label_dir, label_name)
            if os.path.exists(src_label):
                shutil.copy(src_label, f'{DATASET_ROOT}/{split_type}/labels/{label_name}')

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # 2.4 สร้างไฟล์ data.yaml
    yaml_content = f"""
path: {DATASET_ROOT}
train: train/images
val: val/images

names:
  0: object
"""
    with open(os.path.join(DATASET_ROOT, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"✅ เตรียมข้อมูลเสร็จสิ้นที่: {DATASET_ROOT}")

# ==========================================
# 3. ฟังก์ชัน Clean Dataset (ลบรูปที่ไม่มี Label)
# ==========================================
def clean_dataset_folder(images_dir, labels_dir):
    print(f"🧹 กำลังทำความสะอาด: {images_dir}")
    removed_count = 0
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        has_label = False
        if os.path.exists(label_path):
            if os.path.getsize(label_path) > 0:
                has_label = True

        if not has_label:
            os.remove(os.path.join(images_dir, img_file))
            removed_count += 1
            
    print(f"   - ลบรูปไป {removed_count} รูป")

# ==========================================
# 4. Main Loop (สั่งทำงานทุกอย่าง)
# ==========================================
if __name__ == "__main__":
    # 1. เตรียมข้อมูล
    prepare_data()

    # 2. Clean ข้อมูล (Train และ Val)
    clean_dataset_folder(f'{DATASET_ROOT}/train/images', f'{DATASET_ROOT}/train/labels')
    clean_dataset_folder(f'{DATASET_ROOT}/val/images', f'{DATASET_ROOT}/val/labels')

    # 3. เริ่ม Train Model
    print("🏋️‍♂️ เริ่มต้นการเทรนโมเดล...")
    model = YOLO(BASE_MODEL)
    
    results = model.train(
        data=os.path.join(DATASET_ROOT, 'data.yaml'),
        epochs=100,      # ปรับจำนวนรอบตรงนี้
        imgsz=640,       # ขนาดภาพ
        batch=16,        # ถ้า Ram น้อยให้ลดเหลือ 8
        device=0,        # ใช้ GPU (ถ้ามี)
        name='my_seg_model' # ชื่อโฟลเดอร์ผลลัพธ์
    )

    print("🎉 เทรนเสร็จสิ้น!")
    print(f"โมเดลที่ดีที่สุดอยู่ที่: runs/segment/my_seg_model/weights/best.pt")
