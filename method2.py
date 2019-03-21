# import libraries of python OpenCV
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("object detection.mp4")

# creating background subtractor
backsub = cv2.createBackgroundSubtractorMOG2()

white_pixels = []
frame_number = []
i=1

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
    # resize original video to fit into the screen
    scale=0.5
    resized_frames = cv2.resize(frame, (int(scale*1920), int(scale*1080)))

    # apply background subtractor on current frame
    fgmask = backsub.apply(resized_frames, None, -1)
        
    cropped_frame = fgmask[420:450, 300:500]
    
    # counting white pixles
    nonzero = cv2.countNonZero(cropped_frame)
    
    white_pixels.append(nonzero)
    frame_number.append(i)
    i += 1
    
    # original video with the reference area
    cv2.rectangle(resized_frames,(300,420),(500,450),(0,255,0),2)
    cv2.imshow('Frame',resized_frames)
    time.sleep(0.1)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break

plt.plot(frame_number[1:], white_pixels[1:], '.')
plt.show()    