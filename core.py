from ultralytics import YOLO

model=YOLO('/Users/joshua.stanley/Desktop/train32/weights/best.pt')

model.export(format='mlmodel',nms=True, imgsz=640)