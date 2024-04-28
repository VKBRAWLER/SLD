import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

lable_dict = {0: 'paper', 1: 'rock', 2: 'scisor'}

while True:
  data_aux = []
  x_ = []
  y_ = []
  ret, frame = cap.read()
  frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  H, W, _ = frame.shape
  results = hands.process(frame_RGB)

  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
      )
    for hand_landmarks in results.multi_hand_landmarks:
      for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x)
        data_aux.append(y)
        x_.append(x)
        y_.append(y)
    x1 = int(min(x_) * W)
    y1 = int(min(y_) * H)
    x2 = int(max(x_) * W)
    y2 = int(max(y_) * H)
    pridiction = model.predict([np.asarray(data_aux)]) 
    pridiction_symbol = lable_dict[int(pridiction[0])]
    print(pridiction_symbol)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(frame, pridiction_symbol, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3 , (0, 0, 0), 3, cv2.LINE_AA)
  cv2.imshow('frame', frame)
  if cv2.waitKey(25) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows