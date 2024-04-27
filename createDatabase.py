import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []


for dir_ in os.listdir(DATA_DIR):
  print('Class: {}'.format(dir_))
  for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
    data_aux = []
    print('Image: {}'.format(img_path))
    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)
    if results.multi_hand_landmarks is not None:
      for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
          x = hand_landmarks.landmark[i].x
          y = hand_landmarks.landmark[i].y
          data_aux.append(x)
          data_aux.append(y)
      data.append(data_aux)
      labels.append(dir_)
    else:
      print('No hands detected')
f = open('data.pickle', 'wb')
pickle.dump(({'data': data, 'labels': labels}), f)
f.close()