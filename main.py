import cv2 as cv
from util import get_parking_spots_bboxes,empty_or_not
import numpy as np

def calcDiff(im1,im2):
  return np.abs(np.mean(im1)) - np.abs(np.mean(im2))

videoPath = './data/parking_1920_1080.mp4'
mask = './data/mask_1920_1080.png'

mask = cv.imread(mask,0)

cap = cv.VideoCapture(videoPath)

connectedComponents = cv.connectedComponentsWithStats(mask,4,cv.CV_32S)

spots = get_parking_spots_bboxes(connectedComponents)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previousFrame = None

ret = True
step = 30
frame_nmr = 0

while ret:
  ret,frame = cap.read()

  if frame_nmr % step == 0:
    for spot_idx,spot in enumerate(spots):
      x1,y1,w,h = spot
      spot_crop = frame[y1:y1+h,x1:x1+w]
      spot_status = empty_or_not(spot_crop)
      spots_status[spot_idx] = spot_status

  for spot_idx,spot in enumerate(spots):
    spot_status = spots_status[spot_idx] 
    x1,y1,w,h = spots[spot_idx]
    if spot_status:
      frame = cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
    else:
      frame = cv.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
  cv.imshow('frame',frame)
  if cv.waitKey(25) & 0xFF == ord('q'):
    break

  frame_nmr += 1
cap.release()
cv.destroyAllWindows()