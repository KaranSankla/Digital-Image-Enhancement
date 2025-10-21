from ultralytics import YOLO

model = YOLO('/home/karan-sankla/Digital-Image-Enhancement/runs/detect/train10/weights/best.pt')

results = model.predict(
    source='/home/karan-sankla/Downloads',
    imgsz=640,
    save=True,   # saves the predicted images
    conf=0.25,
    device=0,
    show=True
)
