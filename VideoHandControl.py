import cv2
import numpy
import time
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################################
wCam, hCam = 640, 480
####################################
BAR_RANGE = (400, 150)
BLUE = (255, 0, 0)
FROM_RANGE = (30, 300)


def getInRange(value, fromRange, toRange):
    return (toRange[0] + (value - fromRange[0]) * (toRange[1] - toRange[0]) / (fromRange[1] - fromRange[0]))


# Get default audio device using PyCAW
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

## Get current volume range using
# print(volume.GetVolumeRange())
VOL_RANGE = (-65.25, 0.0)
# currentVolumeDb = volume.GetMasterVolumeLevel() #get current volume


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

vText = 0
vBar = 400
pTime = 0
detector = htm.HandDetector(detectionCon=0.8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHand(img)
    listLM, bbox = detector.findPosition(img, showIndx=False)
    if listLM != []:
        x1, y1 = listLM[4][1], listLM[4][2]
        x2, y2 = listLM[8][1], listLM[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        length = int(math.hypot(x1 - x2, y1 - y2))
        if length < 30: length = 30
        if length > 300: length = 300
        vol = getInRange(length, FROM_RANGE, VOL_RANGE)  # for system volume
        vBar = int(getInRange(vol, VOL_RANGE, BAR_RANGE))
        vText = int(getInRange(vol, VOL_RANGE, (0, 100)))
        if length == 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        # print(length, vBar)
        volume.SetMasterVolumeLevel(vol, None)  # set Volume of system

    cv2.rectangle(img, (50, BAR_RANGE[1]), (85, BAR_RANGE[0]), BLUE, 3)
    cv2.rectangle(img, (50, vBar), (85, BAR_RANGE[0]), BLUE, cv2.FILLED)
    cv2.putText(img, f"{vText}%", (40, 450), cv2.FONT_HERSHEY_PLAIN,
                2, BLUE, 2)
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
