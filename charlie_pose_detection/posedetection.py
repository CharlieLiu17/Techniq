import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# the connections made between coordinates
body_lengths = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],\
  [11,12],[11,13],[13,15],[15,17],[17,19],[15,19],[15,21],[12,14],[14,16],\
  [16,22],[16,18],[18,20],[16,20],[12,24],[11,23],[23,24],[24,26],\
  [26,28],[28,30],[28,32],[32,30],[25,23],[27,25],[27,29],[31,27],[29,31]]

body_connections = [[29,31],[31,27],[27,25],[25,23],[23,24],[24,26],[26,28],[28,30],[30,32],[24,12],[12,11],[11,13],[13,15],[15,17],[17,19],[15,21],[12,14],[14,16],[16,22],[16,20],[20,18]]

class pose_detection:
  #put in the userImage and proImage
  def __init__(self, userImage, proImage):
    self.IMAGE_FILES = [userImage, proImage]
    self.landmarks_array = []
    self.transformCode = []
    self.world_landmarks = []

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
        self.world_landmarks.insert(idx, results.pose_world_landmarks)
        # Plot pose world landmarks. PUT IN ANOTHER FUNCTION
        # mp_drawing.plot_landmarks(
        #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

  def show(self, idx):
    # Plot pose world landmarks. PUT IN ANOTHER FUNCTION
    mp_drawing.plot_landmarks(
        self.world_landmarks[idx], mp_pose.POSE_CONNECTIONS)
    print(self.)
    
  def scale(self):
    user = self.landmarks_array[0].landmark
    pro = self.landmarks_array[1].landmark
    shin = body_connections[6]
    userShin = math.sqrt(math.pow(user[shin[0]].x - user[shin[1]].x, 2) + math.pow(user[shin[0]].y - user[shin[1]].y, 2))
    proShin = math.sqrt(math.pow(pro[shin[0]].x - pro[shin[1]].x, 2) + math.pow(pro[shin[0]].y - pro[shin[1]].y, 2))
    ratio = proShin / userShin
    
    body_vectors = []
    for idx, length in enumerate(body_connections):
      vector = self.get_vector(user[length[0]], user[length[1]])
      body_vectors.insert(idx, vector)
      print("pre: " + str(user[length[1]].x))
      user[length[1]].x = user[length[0]].x + ratio * vector[0]
      print("post: " + str(user[length[1]].x))
      user[length[1]].y = user[length[0]].y + ratio * vector[1]
      user[length[1]].z = user[length[0]].z + ratio * vector[2]
    self.landmarks_array[0] = user
    
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
pd = pose_detection("./jump1.jpg", "./jump2.jpg")
pd.detect_pose()
pd.show(0)
pd.scale()
pd.show(0)


#def video_upload_1 () : 
    # upload the first video 