from ultralytics import YOLO
import cv2
import numpy as np
import torch
print(torch.backends.mps.is_available())
model = YOLO('yolov8n.yaml')

results = model.train(data='/Users/joshua.stanley/Desktop/Science Research/config.yaml', epochs=100, save=True, device='mps')