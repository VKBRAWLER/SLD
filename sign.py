import cv2
import os
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

DATA_DIR = './Test'
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
name = 0
while True:
  ret, frame = cap.read()
  cv2.putText(frame, 'Press "s" to save the image for {}'.format(name), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('s'):
    cv2.imwrite(os.path.join(DATA_DIR, '{}.jpg'.format(name)), frame)
    name += 1
    print('Image saved for {}'.format(name))
  elif cv2.waitKey(1) & 0xFF == 27:
    break