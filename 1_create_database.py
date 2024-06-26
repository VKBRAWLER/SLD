import os
import cv2
import mediapipe as mp
import pickle
import tqdm
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

DATA_DIR = './Data'

data = []
labels = []
notDetected = []
index = -1
lable_dict = {}
for className in os.listdir(DATA_DIR):
  index += 1
  for img_path in tqdm.tqdm(os.listdir(os.path.join(DATA_DIR, className))):
    data_aux = []
    x_ = []
    y_ = []
    img = cv2.imread(os.path.join(DATA_DIR, className, img_path))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)
    lable_dict[index] = className
    if results.multi_hand_landmarks is not None:
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

      data.append(data_aux)
      labels.append(index)
    else:
      notDetected.append(os.path.join(DATA_DIR, className, img_path))
  print('Class: {} Completed'.format(className))
if not os.path.exists('./Odata'):
  os.makedirs('./Odata')
with open('./Odata/raw_data.pickle', 'wb') as f:
  pickle.dump({'data': data, 'labels': labels}, f)
print('Data saved in Odata/raw_data.pickle')
if len(notDetected) > 0:
  print('The following images were not detected:')
  for i in notDetected:
    print(i)
print('label_dict: {}'.format(lable_dict))

# Store lable_dict in a file
with open('./Odata/label_dict.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  for key in lable_dict.values():
    writer.writerow([key])
  
print('label_dict saved in Odata/label_dict.csv')