import cv2
import numpy as np

cap = cv2.VideoCapture(0)

trigger = True
counter = 0
width = int(cap.get(3))
heigh = int(cap.get(4))

while(True):
    _, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.rectangle(img, (round(width/4), round(heigh/4)), ( width - round(width/4), heigh - round(heigh/4)), (0, 0, 255), 2)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if(k == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()


