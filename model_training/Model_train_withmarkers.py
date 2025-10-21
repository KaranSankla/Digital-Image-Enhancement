from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data="ycbwith.yaml",
    epochs=50,
    imgsz=640,
    device=0,
    split=0.7  # 80% train, 20% val (YOLO will split automatically)
)