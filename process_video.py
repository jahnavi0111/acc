import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('train_model.h5')

def preprocess(frame):
  frame = cv2.resize(frame, (250, 250))
  return frame
cap = cv2.VideoCapture('03.avi')

frames = []
while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
      frame = preprocess(frame)
      frames.append(frame)
  else:
    break
cap.release()

if len(frames) == 0:
    raise ValueError("Error: No frames extracted from video")

predictions = model.predict(np.array(frames))

print(predictions)
if np.any(predictions == 1):
  print("Accident detected") 
else:
  print("No accident detected")