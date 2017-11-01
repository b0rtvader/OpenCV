import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hand_cascade = cv2.CascadeClassifier('hand1.xml')

    hands = hand_cascade.detectMultiScale(gray, 1.2, 4)

    for (x,y,w,h) in hands:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
