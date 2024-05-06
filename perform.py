import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
import keyboard
from math import hypot

model_dict = pickle.load(open('./Fdata/model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

def pridict(results):
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

def get_distance(frame, results):
  landmarkList = []
  if results.multi_hand_landmarks:
    for handlm in results.multi_hand_landmarks:
      for idx, found_landmark in  enumerate(handlm.landmark):
        print(found_landmark)
        height, width, _ = frame.shape
        x, y = int(found_landmark.x * width), int(found_landmark.y * height)
        if idx == 4 or idx == 8:
          landmark = [idx, x, y]
          landmarkList.append(landmark)
  if len(landmarkList) < 2:
    return
  (x1, y1), (x2, y2) = (landmarkList[0][1], landmarkList[0][2]), (landmarkList[1][1], landmarkList[1][2])
  L = hypot(x2 - x1, y2 - y1)
  return L

def perform(sign):
  if sign == 'point':
    print('point')
    keyboard.press('o') # attack
    time.sleep(0.5)
    keyboard.release('o') # attack
  elif sign == 'open':
    print('open')
    keyboard.press('p') # defend
    time.sleep(0.5)
    keyboard.release('p') # defend
  elif sign == 'yo':
    print('yo')
    keyboard.press('shift') # step
    time.sleep(0.5)
    keyboard.release('shift') # step
  elif sign == 'l':
    print('l')
    keyboard.press('w') # move forward
    time.sleep(0.5)
    keyboard.release('w') # move forward
  elif sign == 'rock':
    print('rock')
    keyboard.press('space') # jump 
    time.sleep(0.5)
    keyboard.release('space') # jump 
  elif sign == 'thumb':
    print('thumb')
    keyboard.press('l') # lock in
    time.sleep(0.5)
    keyboard.release('l') # lock in

labels_dict = {0: 'close', 1: 'l', 2: 'open', 3: 'point', 4: 'rock', 5: 'thumb', 6: 'yo'}
working = False
prevSign = None
while True:
  data_aux = []
  x_ = []
  y_ = []
  ret, frame = cap.read()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == 27:
    print(get_distance(frame, results))
  
cap.release()
cv2.destroyAllWindows()