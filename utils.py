import cv2
import numpy as np
from PIL import Image

def load_image_from_upload(uploaded_file):
    """
    แปลงไฟล์จาก Streamlit uploader เป็น OpenCV Image (numpy array)
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) # โหลดเป็น BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # แปลงเป็น RGB เพื่อแสดงผล
    return image

def draw_segmentation_results(original_image, results):
    """
    รับผลลัพธ์จากโมเดล แล้ววาด Mask/Polygon ลงบนภาพเดิม
    """
    # copy ภาพมาเพื่อไม่ให้กระทบต้นฉบับ
    draw_image = original_image.copy()
    mask_overlay = np.zeros_like(draw_image, dtype=np.uint8)
    
    found_objects = False

    # วนลูปดูผลลัพธ์
    for result in results:
        if result.masks is not None:
            found_objects = True
            for seg in result.masks.xy:
                # แปลงจุด Polygon
                points = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))

                # สุ่มสี (หรือจะ fix สีก็ได้)
                color = tuple(np.random.randint(0, 255, 3).tolist())

                # 1. วาดเส้นขอบ
                cv2.polylines(draw_image, [points], isClosed=True, color=color, thickness=2)

                # 2. ถมสีลงใน Mask overlay
                cv2.fillPoly(mask_overlay, [points], color=color)

    # รวมภาพเดิมกับ Mask (ความโปร่งใส 0.5)
    if found_objects:
        output_image = cv2.addWeighted(draw_image, 1, mask_overlay, 0.5, 0)
        return output_image, True
    else:
        return draw_image, False
