import os
import cv2

# creates a directory to store the data
DATA_DIR = './Data'
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

# number of classes and the size of the dataset
name = str(input("Enter the your name please: "))
number_of_classes = int(input("Enter the number of classes: "))
labelList = []
for i in range(number_of_classes):
  labelList.append(str(input("Enter the label for class {}: ".format(i))))
dataset_size = int(input("Enter the size of the dataset: "))

# confirmation before intering training phase
c = str(input("Note that there are {} class/es and the size of the dataset is {}\nWhich concludes in {} number of images. Do you still want to continue? Y/N: ".format(number_of_classes, dataset_size, number_of_classes*dataset_size)))
while True:
  if c == 'N' or c == 'n':
    exit()
  elif c == 'Y' or c == 'y':
    break
  else:
    c = str(input("Please enter a valid input. Y/N: "))

# initiates the webcam full screen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

for j in range(number_of_classes):
  # creating folders/directory for each class
  if not os.path.exists(os.path.join(DATA_DIR, labelList[j])):
    os.makedirs(os.path.join(DATA_DIR, labelList[j]))
  print('Collecting data for class {}'.format(labelList[j]))

  while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Ready? Press "Space Bar" to Start!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord(' '):
      break
  
  # countdown before the training input starts
  num = 3.7
  while num > 0.5:
    ret, frame = cap.read()
    cv2.putText(frame, "Training Input for {} will start in {}".format(labelList[j], int(num)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    num -= 0.035

  # starts storing images in the directory for ~50 images per second
  counter = len(os.listdir(os.path.join(DATA_DIR, labelList[j])))
  while counter < dataset_size:
    ret, frame = cap.read()
    cv2.putText(frame, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(20)
    cv2.imwrite(os.path.join(DATA_DIR, labelList[j], '{}_{}_{}.jpg'.format(counter,name,labelList[j])), frame)
    counter += 1
cap.release()
cv2.destroyAllWindows()