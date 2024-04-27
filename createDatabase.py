import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
  print('Class: {}'.format(dir_))
  for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
    print('Image: {}'.format(img_path))
    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        img_RGB, # image to draw
        hand_landmarks, # model output
        mp_hands.HAND_CONNECTIONS, # hand connections
        mp_drawing_styles.get_default_hand_landmarks_style(), # drawing styles
        mp_drawing_styles.get_default_hand_connections_style() # drawing styles
      )
      for i in range(len(hand_landmarks.landmark)):
        print(i, hand_landmarks.landmark[i])

    plt.figure()
    plt.imshow(img_RGB)
plt.show()