# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:35:00 2021

@author: Krishna Pavan
"""

import cv2
import numpy as np
from PIL import Image
import os


#setting up path for dataset
path = 'dataset'
recognizer = cv2.face_LBPHFaceRecognizer.create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

#create a function to get images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids
print("\n Training Faces. It may take few seconds. Kindly Wait...")

faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml

recognizer.write('trainer/trainer.yml')

#print no.of faces trained
print("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))

