import cv2
import numpy
import time
import HandTrackingModule as htm

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.HandDetector(detectionCon=0.7)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHand(img)
    lmList, bbox = detector.findPosition(img, showIndx=False)
    if lmList != []:
        fingers = []
        # if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]:
        #     fingers.append(1) #for thumb in up possition
        if (lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]) and (lmList[tipIds[0]][1] > lmList[0][1]):
            fingers.append(1)   #for left hand thumb
        elif (lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]) and (lmList[tipIds[0]][1] < lmList[0][1]):
            fingers.append(1)   #for right hand thumb
        else:
            fingers.append(0)

        for ids in tipIds[1:]:
            if lmList[ids][2] < lmList[ids - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #it will be in form of [thumb, index fniger, ...]
        # print(fingers)
        totalFingers = fingers.count(1)
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 255, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Image")
