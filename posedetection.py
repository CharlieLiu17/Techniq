import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class pose_detection:
  #put in the userImage and proImage
  def __init__(self, userImage, proImage):
    self.IMAGE_FILES = [userImage, proImage]
    self.landmarks_array = []
    self.transformCode = []

  def detect_pose(self):
    # For static images:
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(self.IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
          continue
        self.landmarks_array.insert(idx, results.pose_landmarks)
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', annotated_image)
        #Plot pose world landmarks. PUT IN ANOTHER FUNCTION
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

  #angle_two is pro's
  #angle_one is user's
  def compare_angle(self, angle_one, angle_two):
    return (angle_one > (angle_two - ANGLE_TOLERANCE)) and (angle_one < (angle_two + ANGLE_TOLERANCE))
  
  def transform(self):
    # each index of landmarks array will hold an array of coordinates (landmark)
    # if you want to access the user set of landmarks, it's going to be 0, pro 1
    # want to find a landmark that will serve as the origin and find the difference btwn user and pro of that one coordinate 
    # find the difference and add that to every single coordinate 
    transformCode[0] = self.landmarks_array[1][0].x - self.landmarks_array[0][0].x 
    transformCode[1] = self.landmarks_array[1][0].y - self.landmarks_array[0][0].y

    for bodyPart in self.landmarks_array[0]:
      bodyPart.x = bodyPart.x + transformCode[0]
      bodyPart.y = bodyPart.y + transformCode[1]
      
        
  def get_angle(self, landmark_one, landmark_two, landmark_three):
    vector_one = get_vector(landmark_two, landmark_one)
    vector_two = get_vector(landmark_two, landmark_three)
    cross_prod = vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1] + vector_one[2] * vector_two[2]
    magnitude = math.sqrt(vector_one[0] * vector_one[1] * vector_one[2]) * math.sqrt(vector_two[0] * vector_two[1] * vector_two[2])
    angle = math.acos(cross_prod / magnitude)
    return angle

  def get_vector(self, a, b):
    x = b.x - a.x
    y = b.y - a.y
    z = b.z - a.z
    coordinate_list = [x, y, z]
    return coordinate_list

  def get_2D_vector(self,axis_dropped, a,b): 
    x = b.x - a.x
    y = b.y - a.y
    z = b.z - a.z
    coordinate_list = [x, y, z]
    if axis_dropped == "z": 
      coordinate_list[2] = 0
    elif axis_dropped == "y": 
      coordinate_list[1] = 0
    elif axis_dropped == "x": 
      coordinate_list[0] = 0
    return coordinate_list

  def get_2D_angle(self, axis_dropped, landmark_one, landmark_two, landmark_three): 
    # axis_dropped is the value that will not be considered in the coordinate point 
    # e.g if z is dropped, then the angle between two points in the x-y plane 
    vector_one = self.get_2D_vector(landmark_two, landmark_one)
    vector_two = self.get_2D_vector(landmark_two, landmark_three)
    cross_prod = vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1] + vector_one[2] * vector_two[2]
    magnitude = math.sqrt(vector_one[0] * vector_one[1] * vector_one[2]) * math.sqrt(vector_two[0] * vector_two[1] * vector_two[2])
    angle = math.acos(cross_prod / magnitude)
    return angle
  
  def rotate(self): 
    # find the value for the amount that the user's landmarks need to rotate in order to be in the same orientation to the pros
    # Compare the two coordinates in 3d space, and then find the transformation matrix that transform the user’s image to the professional’s
    # Apply transformation matrix on every coordinate of user
    # rotate about the left hip 

    # find the angle between the two hips with the nose at the origins 
    nose = self.landmarks_array[1][0]
    user_left_hip = self.landmarks_array[0][23]
    pro_left_hip = self.landmarks_array[0][23]
    z_angle = self.get_2D_angle("z",user_left_hip, nose, pro_left_hip)  # angle in the x-y plane- angle of rotation about z axis 
    y_angle = self.get_2D_angle("y",user_left_hip, nose, pro_left_hip)
    x_angle = self.get_2D_angle("x",user_left_hip, nose, pro_left_hip)

    x_rotation_matrix = [[1,0,0],
                        [0,math.cos(x_angle), -math.sin(x_angle)],
                        [0,math.sin(x_angle), math.cos(x_angle)]]
    y_rotation_matrix = [[math.cos(y_angle), 0 ,math.sin(y_angle)],
                        [0, 1, 0],
                        [-math.sin(y_angle),0, math.cos(y_angle)]]
    z_rotation_matrix = [[math.cos(z_angle),-math.sin(z_angle),0],
                        [math.cos(z_angle), -math.sin(z_angle),0],
                        [0,0,1]]
    
    self.landmarks_array[0] *= x_rotation_matrix
    self.landmarks_array[0] *= y_rotation_matrix
    self.landmarks_array[0] *= z_rotation_matrix

    



pd = pose_detection("./jump1.jpg", "./jump2.jpg")
pd.detect_pose()



