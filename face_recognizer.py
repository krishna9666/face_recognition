# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:04:24 2021

@author: Krishna Pavan
"""

import cv2
import numpy as np
import os

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read('trainer/trainer.yml')
cascadepath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadepath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Krishna Pavan', 'Venkatesh', 'Vishal']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 720)


minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor = 1.3,
                                         minNeighbors = 5,
                                         minSize = (int(minW), int(minH)))
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        if(confidence < 100):
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
            
        else:
            id = "unkonwn"
            confidence = "{0}%".format(round(100 - confidence))
            
        cv2.putText(img,
                        str(id),
                        (x+5, y-5),
                         font,
                         1, (255,255,255),2)
       # cv2.putText(img,
        #                str(confidence),
        #                (x+5,y-5),
        #                font,
          #              1,(255,255,0),1) 
    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break
    