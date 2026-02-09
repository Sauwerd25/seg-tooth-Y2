def run_inference(model_wrapper, image, conf_threshold=0.25):
    """
    ฟังก์ชันสำหรับรับภาพและโมเดล เพื่อทำการ Predict
    
    Args:
        model_wrapper: object ของ class YoloSegmentationModel ที่โหลด weight แล้ว
        image: ภาพ numpy array
        conf_threshold: ค่าความมั่นใจ
    
    Returns:
        results: ผลลัพธ์ดิบจาก YOLO
    """
    yolo = model_wrapper.get_model()
    
    if yolo is None:
        raise ValueError("Model has not been loaded yet!")

    # สั่ง Predict (save=False เพื่อไม่ให้เขียนไฟล์ลง disk)
    results = yolo.predict(source=image, save=False, verbose=False, conf=conf_threshold)
    
    return results
