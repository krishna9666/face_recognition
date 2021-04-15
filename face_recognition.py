# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:42:56 2021

@author: Krishna Pavan
"""

import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')

face_id = input('\n Enter user_id')
print("\n Intializing face capture. Please look at the camera and wait...")
count = 0
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 720)


while(True):
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5,
                                        minSize=(20,20))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        cv2.imwrite("dataset/User."+str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('Image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27: 
        break
    elif count >= 50:
        break
cap.release()
cv2.destroyAllWindows()