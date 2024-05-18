import pickle
import time
import cv2
import cv2.text
import mediapipe as mp
import numpy as np
import keyboard
from math import hypot
import csv

model_dict = pickle.load(open('./Odata/model.p', 'rb'))
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
        # print(found_landmark)
        height, width, _ = frame.shape
        x, y = int(found_landmark.x * width), int(found_landmark.y * height)
        if idx == 4 or idx == 8:
          landmark = [idx, x, y]
          landmarkList.append(landmark)
  if len(landmarkList) < 2:
    return 0
  (x1, y1), (x2, y2) = (landmarkList[0][1], landmarkList[0][2]), (landmarkList[1][1], landmarkList[1][2])
  L = hypot(x2 - x1, y2 - y1)
  return L

labels_dict = {}
with open('./Odata/label_dict.csv', 'r') as file:
  reader = csv.reader(file)
  for index, line in enumerate(reader):
    labels_dict[int(index)] = line[0]

prevSign = None
executing = False
key = None
length = 0
last = 0

while True:
  data_aux = []
  x_ = []
  y_ = []
  ret, frame = cap.read()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(frame_rgb)
  sign = pridict(results)
  if key == 'E':
    if sign == 'Q':
      print('Dark Cover Placed')
      keyboard.press_and_release('e')
      executing = False
      key = None
    else:
      new_length = get_distance(frame, results)
      print(new_length)
      if new_length > length:
        print('Farther')
        keyboard.press('w')
        time.sleep(0.1)
        keyboard.release('w')
      elif new_length < length:
        print('Closer')
        keyboard.press('s')
        time.sleep(0.1)
        keyboard.release('s')
      else :
        print('Same')
      time.sleep(0.4)
  if sign == prevSign or sign is None:
    continue
  elif executing:
    if sign == 'P':
      print('canceled')
      keyboard.release('4')
      executing = False
      key = None
      length = 0
      if key == 'LLC':
        keyboard.release('w')
      keyboard.press_and_release('1')
    elif key == 'C' and sign == 'C1':
      print('executed')
      keyboard.press_and_release('w')
      executing = False
      key = None
    elif key == 'Q' and sign == 'Q1':
      print('executed')
      keyboard.press_and_release('w')
      executing = False
      key = None
    elif key == 'X' and sign == 'X1':
      print('executed')
      keyboard.press_and_release('w')
      executing = False
      key = None
    elif key == 'LLC' and sign != 'LLC':
      print('Shooting Stoped')
      keyboard.release('w')
      executing = False
      key = None
  elif sign == 'LC':
    print('shoot')
    keyboard.press_and_release('w')
  elif sign == 'R':
    print('reload')
    keyboard.press_and_release('r')
  elif sign == 'SPACE':
    print('jump')
    keyboard.press_and_release('space')
  elif sign == 'C':
    print('Shrouded Step')
    key = 'C'
    keyboard.press_and_release('c')
    executing = True
  elif sign == 'Q':
    print('Paranoia')
    key = 'Q'
    keyboard.press_and_release('q')
    executing = True
  elif sign == 'X':
    print('From The Shadows')
    key = 'X'
    keyboard.press_and_release('x')
    executing = True
  elif sign == 'LLC':
    print('Shooting')
    key = 'LLC'
    keyboard.press('w')
    executing = True
  elif sign == 'E':
    print("Dark Cover")
    key = 'E'
    length = (get_distance(frame, results)/2)
    print(length)
    keyboard.press_and_release('e')
    executing = True
  elif sign == 'P':
    print('defusing')
    keyboard.press('4')
    executing = True
    key = 'P'
    if key == 'LLC':
      keyboard.release('w')
  prevSign = sign
  # cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  # cv2.imshow('frame', frame)
  if cv2.waitKey(25) & 0xFF == 27:
    break
cap.release()
cv2.destroyAllWindows()