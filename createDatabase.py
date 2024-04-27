import os
import cv2
import matplotlib.pyplot as plt

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
  print('Class: {}'.format(dir_))
  for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
    print('Image: {}'.format(img_path))
    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_RGB)
plt.show()