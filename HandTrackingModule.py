import cv2
# to install: conda install mediapipe
import mediapipe as mp
# import numpy
import time
import math


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
tipIds = [4, 8, 12, 16, 20]

class HandDetector():
    '''we have used mediapipe module in this detector'''

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHands, detectionCon, trackCon)
        self.mpDraws = mp.solutions.drawing_utils

    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraws.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, showIndx=True, drawBox=False):
        '''handNo is for when there is more than one hands in the screen '''
        xList = []
        yList = []
        bbox = []
        self.lmList = []  # landMark list
        # To print id on hand landmark
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)  # x & y are in ration form
                self.lmList.append([id, cx, cy])
                xList.append(cx)
                yList.append(cy)
                if showIndx:
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, BLUE, 1)
            bbox = min(xList), min(yList), max(xList), max(yList)
                    #xmin, ymin, xmax, ymax
            if drawBox:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), GREEN, 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]:
        #     fingers.append(1) #for thumb in up possition
        if (self.lmList[tipIds[0]][1] > self.lmList[tipIds[0] - 1][1]) and (self.lmList[tipIds[0]][1] > self.lmList[0][1]):
            fingers.append(1)  # for left hand thumb
        elif (self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]) and (self.lmList[tipIds[0]][1] < self.lmList[0][1]):
            fingers.append(1)  # for right hand thumb
        else:
            fingers.append(0)

        for ids in tipIds[1:]:
            if self.lmList[ids][2] < self.lmList[ids - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self,img,p1,p2,draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = int(math.hypot(x1 - x2, y1 - y2))
        if length < 30: length = 30
        if length > 300: length = 300

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return img, length, [x1, y1, x2, y2, cx, cy]
        '''
        index id according to fingertip is given below
        indx - fingertip
        4 - thumb
        8 - index finger
        12 - middle finger
        16 - ring finger
        20 - pinky finger
        '''


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHand(img)
        lmList, bbox = detector.findPosition(img,drawBox=True)
        if len(lmList) != 0:
            print(lmList[4])  # use index as given instruction below the function
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    BLUE, 3)
        cv2.imshow("Image ", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
