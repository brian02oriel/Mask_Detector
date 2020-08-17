import numpy as np
import cv2
from joblib import load
from LocalBinaryPattern import LocalBinaryPatterns

cap = cv2.VideoCapture(0)
width = int(cap.get(3))
heigh = int(cap.get(4))
radius = 3
no_points = 8 * radius

desc = LocalBinaryPatterns(no_points, radius)
decoding = {
    0: 'No Mask',
    1: 'With Mask'
}
while(True):
    _, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.rectangle(img, (round(width/4), round(heigh/8)), (width - round(width/4), heigh - round(heigh/8)), (0, 0, 255), 2)
    roi = img[round(heigh/8): heigh - round(heigh/8), round(width/4): width - round(width/4)]
    roi = cv2.resize(roi, (200, 200))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(roi)
    hist = np.array(hist)
    model = load('classifier.joblib')
    results = model.predict(hist.reshape(1, -1))
    print(decoding[results[0]])
    if(results[0] == 1):
        cv2.rectangle(img, (round(width/4), round(heigh/8)), (width - round(width/4), heigh - round(heigh/8)), (0, 255, 0), 3)
        cv2.rectangle(img, (round(width/4) - 2, round(heigh/8) + heigh), (width - round(width/4) + 2, heigh - round(heigh/8)), (0, 255, 0), -1)
        cv2.putText(img, decoding[results[0]], (round(width/4) + round(width/8), heigh - round(heigh/16)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Face Recognition', img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

