import cv2
import cvlib as cv
import numpy as np
from os import listdir
from os.path import isfile, join

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

# open webcam
webcam = cv2.VideoCapture(0)

sample_num = 0   

while webcam.isOpened():

    ret, frame = webcam.read()
    sample_num = sample_num + 1
    
    if not ret:
        print("Could not read frame")
        exit()
        
    face, confidence = cv.detect_face(frame)

    try:
        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            Y = startY - 10 if startY - 10 > 10 else startY + 10
    
            if sample_num % 8  == 0:
                face_in_img = frame[startY:endY, startX:endX, :]
                face_in_img = cv2.cvtColor(face_in_img, cv2.COLOR_BGR2GRAY)
                result = model.predict(face_in_img)
                confidence = int(100*(1-(result[1])/300))

                if confidence > 75:
                    cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    text = str(confidence)+'% Confidence it is target'
                    cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow('Face Cropper', frame)

                else:
                    roi = frame[startY:endY, startX:endX] # 관심영역 지정
                    roi = cv2.GaussianBlur(roi, (0, 0), 3) # 블러(모자이크) 처리
                    frame[startY:endY, startX:endX] = roi 
                    cv2.imshow('Face Cropper', frame)


    except:
        cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', frame)
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()
