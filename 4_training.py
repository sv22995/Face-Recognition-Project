import cv2
import os
import numpy as np
from PIL import Image

dataset_path = "dataset/"
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Wait for the training to be completed...")

def getImagesAndLabels(dataset_path):

    paths = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path)]     
    Samples , IDs = [] , []

    for image_path in paths:

        PIL_img = Image.open(image_path).convert('L') 
        img_numpy = np.array(PIL_img,"uint8")

        ID = int(os.path.split(image_path)[-1].split(".")[1])
        Coordinates = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in Coordinates:
            Samples.append(img_numpy[y:y+h,x:x+w])
            IDs.append(ID)

    return Samples , IDs

Coordinates , IDs = getImagesAndLabels(dataset_path)

recognizer.train(Coordinates , np.array(IDs))
recognizer.write("trainer/trainer.yml")

print("TRAINING COMPLETED!")
print("\n {0} face trained".format(len(np.unique(IDs))))