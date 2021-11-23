# import cv2
# import os
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# def synchronize(user_path, pro_path, body_part_flag, saved_mp_data):
#     for root, dirs, files in os.walk(pro_path):
#         for file in files:
#             if file.endswith('.jpg'):
#                 image = cv2.imread(file)
#         image_height, image_width, _ = image.shape
#         # Convert the BGR image to RGB before processing.
#         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# synchronize("./vid_extract_frames/user", "./vid_extract_frames/pro", 1)