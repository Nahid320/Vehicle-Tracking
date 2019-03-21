# OpenCV Python program to detect cars in video frame 
# import libraries of python OpenCV  
import cv2 
import time
import numpy as np

# capture frames from a video and count the number of frames
cap = cv2.VideoCapture('object detection.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# load XML car classifiers 
car_cascade = cv2.CascadeClassifier('cars.xml') 

sum=0

# loop runs if capturing has been initialized. 
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # resize original video to fit into the screen
        scale=0.5
        resized_frames = cv2.resize(frame, (int(scale*1920), int(scale*1080)))
     
        # Detects cars of different sizes in the input image 
        cars = car_cascade.detectMultiScale(resized_frames, 1.1, 1, 0, (60,60), (100,100))

        # count the number of detected cars in each frame 
        if type(cars) == np.ndarray:
            count=cars.size/4 
            sum += count
            
        # To draw a rectangle around each car 
        for (x,y,w,h) in cars: 
            cv2.rectangle(resized_frames,(x,y),(x+w,y+h),(0,0,255),2) 
            
            # Display frames in a window  
            cv2.putText(resized_frames,'COUNT IN EACH FRAME : %r' %int(count), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('video2', resized_frames)
            time.sleep(0.1)
     
        # Wait for Esc key to stop 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:   
        break
        
total = int(np.ceil(sum/frame_count))
print("total count = ", total)
time.sleep(2)

# De-allocate any associated memory usage 
cv2.destroyAllWindows()