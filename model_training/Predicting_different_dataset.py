from ultralytics import YOLO

# Load your trained model (use the best weights)
model = YOLO('runs/detect/train/weights/best.pt')  # or 'last.pt'

results = model.predict(
    source='path/to/test/images/',
    conf=0.25,
    device=0,
    save=True
)
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print(f"Detected {len(boxes)} objects")

    # Get detection details
    for box in boxes:
        cls = int(box.cls[0])  # class id
        conf = float(box.conf[0])  # confidence
        xyxy = box.xyxy[0].cpu().numpy()  # bbox coordinates
        print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: {xyxy}")