from ultralytics import YOLO

class YoloSegmentationModel:
    def __init__(self):
        """
        สร้างโครงไว้ ตัวแปร model เริ่มต้นเป็น None
        """
        self.model = None

    def load_weights(self, weights_path):
        """
        โหลดไฟล์ .pt เข้าสู่ระบบ
        """
        print(f"Loading weights from: {weights_path}")
        try:
            self.model = YOLO(weights_path)
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def get_model(self):
        return self.model
