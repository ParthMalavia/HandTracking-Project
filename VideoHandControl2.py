import cv2
# import numpy
import time
import HandTrackingModule as htm
# import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################################
wCam, hCam = 640, 480
####################################

def getInRange(value, fromRange, toRange):
    return toRange[0] + (value - fromRange[0]) * (toRange[1] - toRange[0]) / (fromRange[1] - fromRange[0])


# Get default audio device using PyCAW
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# currentVolumeDb = volume.GetMasterVolumeLevel() #get current volume

BAR_RANGE = (400, 150)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
FROM_RANGE = (30, 300)
vPer = 0
vBar = 400
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionCon=0.8, maxHands=1)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHand(img)
    lmList, bbox = detector.findPosition(img, showIndx=False)
    if lmList != []:
        # Filter based on area
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 300 < area < 1000:
            # find distance between indx and thumb
            img, length, lineInfo = detector.findDistance(img, 4, 8)

            # convert volume
            vBar = int(getInRange(length, FROM_RANGE, BAR_RANGE))
            vPer = int(getInRange(length, FROM_RANGE, (0, 100)))

            # reducce resolition to make it smoother
            smoothness = 5
            vPer = smoothness * int(vPer / smoothness)

            # check fingers up
            fingers = detector.fingersUp()
            # if pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(vPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, GREEN, cv2.FILLED)

    cv2.rectangle(img, (50, BAR_RANGE[1]), (85, BAR_RANGE[0]), BLUE, 3)
    cv2.rectangle(img, (50, vBar), (85, BAR_RANGE[0]), BLUE, cv2.FILLED)
    cv2.putText(img, f"{vPer}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, BLUE, 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                2, BLUE, 2)
    cv2.imshow("Image ", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
