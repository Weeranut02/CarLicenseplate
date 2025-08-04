from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("test.jpg")

# แสดงภาพพร้อมกล่อง
results[0].show()
 