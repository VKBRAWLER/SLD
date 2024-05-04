import pickle

import cv2
import mediapipe as mp
import numpy as np
import keyboard

model_dict = pickle.load(open('./Fdata/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

def pridict(frame):
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))
    prediction = model.predict([np.asarray(data_aux[:42])])  # Limit the input data to 42 features
    predicted_character = labels_dict[int(prediction[0])]
    return(predicted_character)

def perform(sign):
  print(sign)
  while True:
    keyboard.press('space')
    ret, frame = cap.read()
    if pridict(frame) != sign:
      break
    if cv2.waitKey(20) & 0xFF == 27:
      break
  keyboard.release('space')

labels_dict = {0: 'close', 1: 'l', 2: 'open', 3: 'point', 4: 'rock', 5: 'thumb', 6: 'yo'}
while True:
  data_aux = []
  x_ = []
  y_ = []
  ret, frame = cap.read()
  sign = pridict(frame)
  if sign == None:
    continue
  else:
    perform(sign)
  if cv2.waitKey(20) & 0xFF == 27:
    break
cap.release()
cv2.destroyAllWindows()