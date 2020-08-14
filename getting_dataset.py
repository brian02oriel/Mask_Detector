import cv2
import numpy as np

cap = cv2.VideoCapture(0)

trigger = False
counter = 0
width = int(cap.get(3))
heigh = int(cap.get(4))
class_name = ''

while(True):
    _, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.rectangle(img, (round(width/4), round(heigh/8)), (width - round(width/4), heigh - round(heigh/8)), (0, 0, 255), 2)
    
    if(counter == 200):
        trigger = not trigger
        counter = 0
    
    if(trigger):
        roi = img[round(heigh/8): heigh - round(heigh/8), round(width/4): width - round(width/4)]
        roi = cv2.resize(roi, (200, 200))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('dataset/'+ class_name + '/' + str(counter) + '.jpg', roi)
        counter += 1
        text = "Collected Samples of {}: {}".format(class_name, counter)
        cv2.putText(img, text, (round(width/4), heigh - round(heigh/8) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        text1 = "Press 'p' to collecting mask class and press 'n' to collecting faces without mask"
        text2 = "Press 'q' to quit"
        cv2.putText(img, text1, (round(width/4), heigh - round(heigh/8) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text2, (round(width/4), heigh - round(heigh/8) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('img', img)
    k = cv2.waitKey(1)

    # If user press 'p' save the examples for people with a mask
    if k == ord('p'):
        trigger = not trigger
        class_name = 'positive'
    
    # If user press 'n' save the examples for people without a mask
    if k == ord('n'):
        trigger = not trigger
        class_name = 'negative'
    

    if(k == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()


