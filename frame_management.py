import cv2
import time
import math
import os
import mediapipe as mp

mp_pose = mp.solutions.pose
REFRESH_RATE = 0.125
def extract_frame(vid_path, category):
    # Opens the Video file
    cap = cv2.VideoCapture(vid_path)

    #Find FPS
    fps = cap.get(cv2.CAP_PROP_FPS);

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width = len(str(int(frames)))

    frame_refresh_int = round(fps * REFRESH_RATE)
    i = 0  
    count = 0
    print(width)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # if (i % frame_refresh_int == 0):
        cv2.imwrite('./vid_extract_frames/' + category + '/frame'+ str(i).zfill(width) +'.jpg', frame)
            # count += 1
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()


extract_frame("./test_inputs/video/charlie2vid.mp4", 'pro')