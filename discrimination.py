# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image


model = load_model('model.h5')
model.summary()

print("원하는 이모지를 선택하세요!")
flag=input("1. 기쁨 2 슬픔 3. 하트 4. 생일 5. 왕관 ==> ")

if flag=='1':
    src2=cv2.imread('img/smile.png',-1)
elif flag=='2':
    src2=cv2.imread('img/sad.png',-1)
elif flag=='3':
    src2=cv2.imread('img/heart.png',-1)
elif flag=='4':
    src2=cv2.imread('img/birthday.png',-1)
elif flag=='5':
    src2=cv2.imread('img/crown.png',-1)


def transparent_overlay(src ,overlay ,pos=(0,0) ,scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image

    if flag=='4': 
        y, x = pos[0], pos[1]-(h+2)    # Position of foreground/overlay image
    elif flag=='5':
        y, x = pos[0], pos[1]-h-2
    else:
        y, x = pos[0], pos[1]   

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0) # read the alpha channel 
            src[x + i][y + j] = alpha*overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]

    return src


# open webcam
webcam = cv2.VideoCapture(0)


if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            face_region = frame[startY:endY, startX:endX]
            
            face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
            
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            prediction = model.predict(x)
 
            if prediction >= 0.7: # 타켓 판별
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "target ({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            else: # 논타켓 판별
                roi = frame[startY:endY, startX:endX] # 관심영역 지정
                roi = cv2.GaussianBlur(roi, (0, 0), 3) # 블러(모자이크) 처리
                frame[startY:endY, startX:endX] = roi 
                src = cv2.resize(src2, dsize=(endX - startX,(endY - startY)), interpolation=cv2.INTER_AREA)
                frame = transparent_overlay(frame, src, (startX, startY))
                
    # display output
    cv2.imshow("target classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows() 