import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
import pickle
import numpy as np
import keyboard
import time
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
lable_dict = {0: 'CLOSE', 1: 'OPEN', 2: 'PEACE'}

def pridict(frame):
  data_aux = []
  frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_RGB)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x)
        data_aux.append(y)
    pridiction = model.predict([np.asarray(data_aux)]) 
    pridiction_symbol = lable_dict[int(pridiction[0])]
    return pridiction_symbol

while True:
  ret, frame = cap.read()
  prdidiction_symbol = pridict(frame)
  if prdidiction_symbol == 'OPEN':
    print('OPEN')
    keyboard.press('p')
    time.sleep(1)
    keyboard.release('p')
  elif prdidiction_symbol == 'PEACE':
    print('PEACE')
    keyboard.press('o')
    time.sleep(0.1)
    keyboard.release('o')
  
  # cv2.imshow('frame', frame)
  if cv2.waitKey(25) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()