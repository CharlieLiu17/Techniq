import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



ANGLE_TOLERANCE = math.radians(15)
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
    self.transformCode = [None, None, None]
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

  def double_show(self):
        image = cv2.imread('./tmp/annotated_image' + str(0) + '.png')
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # # bilateral filter to "results.segmentation_mask" with "image".
    # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = BG_COLOR
    # annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            image,
            self.landmarks_array[1],
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('./tmp/annotated_image' + str(1) + '.png', image)
        # arr = [11, 13, 15]
        # for val in arr:
        #     print(str(val) + ".x: " + str(self.landmarks_array[0].landmark[val].x))
        #     print(str(val) + ".y: " + str(self.landmarks_array[0].landmark[val].y))
        #     print(str(val) + ".z: " + str(self.landmarks_array[0].landmark[val].z))
    
  def scale(self):
    # user = self.landmarks_array[0].landmark
    # pro = self.landmarks_array[1].landmark
    shin = body_connections[6]
    userShin = math.sqrt(math.pow(self.landmarks_array[0].landmark[shin[0]].x - self.landmarks_array[0].landmark[shin[1]].x, 2) + math.pow(self.landmarks_array[0].landmark[shin[0]].y - self.landmarks_array[0].landmark[shin[1]].y, 2))
    proShin = math.sqrt(math.pow(self.landmarks_array[1].landmark[shin[0]].x - self.landmarks_array[1].landmark[shin[1]].x, 2) + math.pow(self.landmarks_array[1].landmark[shin[0]].y - self.landmarks_array[1].landmark[shin[1]].y, 2))
    ratio = proShin / userShin
    
    body_vectors = []
    for idx, length in enumerate(body_connections):
      vector = self.get_vector(self.landmarks_array[0].landmark[length[0]], self.landmarks_array[0].landmark[length[1]])
      body_vectors.insert(idx, vector)
      self.landmarks_array[0].landmark[length[1]].x = self.landmarks_array[0].landmark[length[0]].x + ratio * vector[0]
      self.landmarks_array[0].landmark[length[1]].y = self.landmarks_array[0].landmark[length[0]].y + ratio * vector[1]
      self.landmarks_array[0].landmark[length[1]].z = self.landmarks_array[0].landmark[length[0]].z + ratio * vector[2]
    # self.landmarks_array[0] = user
    
    
  #angle_two is pro's
  #angle_one is user's
  def compare_angle(self, angle_one, angle_two):
    if (angle_one > (angle_two + ANGLE_TOLERANCE)):
      return 1
    elif (angle_one < (angle_two - ANGLE_TOLERANCE)):
      return -1
    else:
      return 0
  
  def transform(self):
        # each index of landmarks array will hold an array of coordinates (landmark)
    # if you want to access the user set of landmarks, it's going to be 0, pro 1
    # want to find a landmark that will serve as the origin and find the difference btwn user and pro of that one coordinate 
    # find the difference and add that to every single coordinate 
    
    self.transformCode[0] = self.landmarks_array[1].landmark[29].x - self.landmarks_array[0].landmark[29].x 
    self.transformCode[1] = self.landmarks_array[1].landmark[29].y - self.landmarks_array[0].landmark[29].y
    self.transformCode[2] = self.landmarks_array[1].landmark[29].z - self.landmarks_array[0].landmark[29].z

    for bodyPart in self.landmarks_array[0].landmark:
      bodyPart.x = bodyPart.x + self.transformCode[0]
      bodyPart.y = bodyPart.y + self.transformCode[1]
      bodyPart.z = bodyPart.z + self.transformCode[2]
  
  def body_check(self):
    specBodyPart = {}
    print("Please wait while the advice on your form is being generated!")
    #arm
    specBodyPart["armUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[11], self.landmarks_array[0].landmark[13], self.landmarks_array[0].landmark[15])
    specBodyPart["armProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[11], self.landmarks_array[1].landmark[13], self.landmarks_array[1].landmark[15])
    specBodyPart["armUserRight"] = self.get_2D_angle(self.landmarks_array[0].landmark[12], self.landmarks_array[0].landmark[14], self.landmarks_array[0].landmark[16])
    print(math.degrees(specBodyPart["armUserRight"]))
    specBodyPart["armProRight"] = self.get_2D_angle(self.landmarks_array[1].landmark[12], self.landmarks_array[1].landmark[14], self.landmarks_array[1].landmark[16])
    print(math.degrees(specBodyPart["armProRight"]))


    # left arm
    if (self.compare_angle(specBodyPart["armUserLeft"], specBodyPart["armProLeft"]) == -1):
        print("You should extend your left elbow out more around " + str(round(math.degrees(specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["armUserLeft"], specBodyPart["armProLeft"]) == 1):
        print("Your left elbow is extended too far. You should contract your left elbow more by " + str(round(math.degrees(specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]))) + " degrees.")
    # else:
        # print("Your left elbow position seems great! Keep up the good work.")

    # right arm
    if (self.compare_angle(specBodyPart["armUserRight"], specBodyPart["armProRight"]) == -1):
        print("You should extend your right elbow out more around " + str(round(math.degrees(specBodyPart["armProRight"] - specBodyPart["armUserRight"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["armUserRight"], specBodyPart["armProRight"]) == 1):
        print("Your right elbow is extended too far. You should contract your right elbow more by " + str(round(math.degrees(specBodyPart["armProRight"] - specBodyPart["armUserRight"]))) + " degrees.")
    # else:
        # print("Your right elbow position seems great! Keep up the good work.")
    

    #leg/knee
    specBodyPart["kneeUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[23], self.landmarks_array[0].landmark[25], self.landmarks_array[0].landmark[27])
    specBodyPart["kneeProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[23], self.landmarks_array[1].landmark[25], self.landmarks_array[1].landmark[27])
    specBodyPart["kneeUserRight"] = self.get_2D_angle(self.landmarks_array[0].landmark[24], self.landmarks_array[0].landmark[26], self.landmarks_array[0].landmark[28])
    specBodyPart["kneeProRight"] = self.get_2D_angle(self.landmarks_array[1].landmark[24], self.landmarks_array[1].landmark[26], self.landmarks_array[1].landmark[28])
    if (self.compare_angle(specBodyPart["kneeUserLeft"], specBodyPart["kneeProLeft"]) == -1):
        print("You should extend your left knee out more around " + str(round(math.degrees(specBodyPart["kneeProLeft"] - specBodyPart["kneeUserLeft"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["kneeUserLeft"], specBodyPart["kneeProLeft"]) == 1):
        print("Your left knee is extended too far. You should contract your left knee  by " + str(round(math.degrees(specBodyPart["kneeProLeft"] - specBodyPart["kneeUserLeft"]))) + " degrees.")
    # else:
        # print("Your left knee position seems great! Keep up the good work.")
    
    if (self.compare_angle(specBodyPart["kneeProRight"], specBodyPart["kneeUserRight"]) == -1):
        print("You should extend your right knee out more around " + str(round(math.degrees(specBodyPart["kneeProRight"] - specBodyPart["kneeUserRight"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["kneeProRight"], specBodyPart["kneeUserRight"]) == 1):
        print("Your right knee is extended too far. You should contract your right knee. Contract it closer to your body by " + str(round(math.degrees(specBodyPart["kneeProRight"] - specBodyPart["kneeUserRight"]))) + " degrees.")
    # else:
        # print("Your right knee position seems great! Keep up the good work.")
    

    #hip
    specBodyPart["hipUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[24], self.landmarks_array[0].landmark[23], self.landmarks_array[0].landmark[25])
    specBodyPart["hipProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[24], self.landmarks_array[1].landmark[23], self.landmarks_array[1].landmark[25])
    specBodyPart["hipUserRight"] = self.get_2D_angle(self.landmarks_array[0].landmark[23], self.landmarks_array[0].landmark[24], self.landmarks_array[0].landmark[26])
    specBodyPart["hipProRight"] = self.get_2D_angle(self.landmarks_array[1].landmark[23], self.landmarks_array[1].landmark[24], self.landmarks_array[1].landmark[26])

    if (self.compare_angle(specBodyPart["hipUserLeft"], specBodyPart["hipProLeft"]) == -1):
        print("You should extend your left leg out more by around " + str(round(math.degrees(specBodyPart["hipProLeft"] - specBodyPart["hipUserLeft"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["hipUserLeft"], specBodyPart["hipProLeft"] == 1)):
        print("Your left leg is extended too far outwards. You should bring in your left leg by " + str(round(math.degrees(specBodyPart["hipProLeft"] - specBodyPart["hipUserLeft"]))) + " degrees.")
    # else:
        # print("Your leg position seems great! Keep up the good work.")
    

    if (self.compare_angle(specBodyPart["hipUserRight"], specBodyPart["hipProRight"]) == -1):
        print("You should extend your right leg out more by around " + str(round(math.degrees(specBodyPart["hipProRight"] - specBodyPart["hipUserRight"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["hipUserRight"], specBodyPart["hipProRight"]) == 1):
        print("Your right leg is extended too far outwards. You should bring your right leg in by " + str(round(math.degrees(specBodyPart["hipProRight"] - specBodyPart["hipUserRight"]))) + " degrees.")
    # else:
        # print("Your right leg position seems great! Keep up the good work.")


    #side body
    specBodyPart["sideUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[11], self.landmarks_array[0].landmark[23], self.landmarks_array[0].landmark[25])
    specBodyPart["sideProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[11], self.landmarks_array[1].landmark[23], self.landmarks_array[1].landmark[25])

    if (self.compare_angle(specBodyPart["sideUserLeft"], specBodyPart["sideProLeft"]) == -1):
        print("You should drop your left shoulder and lean in further by " + str(round(math.degrees(specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]))) + " degrees.")
    elif (self.compare_angle(specBodyPart["sideUserLeft"], specBodyPart["sideProLeft"]) == 1):
        print("You should drop your right shoulder and lean in further by " + str(round(math.degrees(specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]))) + " degrees.")
    # else:
        # print("Your left oblique side bend seems great! Keep up the good work.")


    #arm pit
    specBodyPart["armPitUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[13], self.landmarks_array[0].landmark[11], self.landmarks_array[0].landmark[23])
    specBodyPart["armPitProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[13], self.landmarks_array[1].landmark[11], self.landmarks_array[1].landmark[23])
    specBodyPart["armPitUserRight"] = self.get_2D_angle(self.landmarks_array[0].landmark[14], self.landmarks_array[0].landmark[12], self.landmarks_array[0].landmark[24])
    specBodyPart["armPitProRight"] = self.get_2D_angle(self.landmarks_array[1].landmark[14], self.landmarks_array[1].landmark[12], self.landmarks_array[1].landmark[24])

    if (self.compare_angle(specBodyPart["armPitUserLeft"], specBodyPart["armPitProLeft"]) == -1):
        print("Your left arm is dropped to low to your side. Your left arm should be raised up more by " + str(round(math.degrees(specBodyPart["armPitProLeft"] - specBodyPart["armPitUserLeft"]))) + " degrees.")
    elif (self.compare_angle(specBodyPart["armPitUserLeft"], specBodyPart["armPitProLeft"]) == 1):
        print("Your left arm is raised to high. Your left arm should be lowered down to your side more by " + str(round(math.degrees(specBodyPart["armPitProLeft"] - specBodyPart["armPitUserLeft"]))) + " degrees.")
    # else:
        # print("Your left arm position looks great! Keep up the good work.")

    if (self.compare_angle(specBodyPart["armPitUserRight"], specBodyPart["armPitProRight"]) == -1):
        print("Your right arm is dropped to low to your side. Your right arm should be raised up more by " + str(round(math.degrees(specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["armPitUserRight"], specBodyPart["armPitProRight"]) == 1):
        print("Your right arm is raised to high. Your right arm should be lowered down to your side more by " + str(round(math.degrees(specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]))) + " degrees.")
    # else:
        # print("Your right arm position looks great! Keep up the good work.")


    #side body
    specBodyPart["sideUserLeft"] = self.get_2D_angle(self.landmarks_array[0].landmark[11], self.landmarks_array[0].landmark[23], self.landmarks_array[0].landmark[25])
    specBodyPart["sideProLeft"] = self.get_2D_angle(self.landmarks_array[1].landmark[11], self.landmarks_array[1].landmark[23], self.landmarks_array[1].landmark[25])
    specBodyPart["sideUserRight"] = self.get_2D_angle(self.landmarks_array[0].landmark[12], self.landmarks_array[0].landmark[24], self.landmarks_array[0].landmark[26])
    specBodyPart["sideProRight"] = self.get_2D_angle(self.landmarks_array[1].landmark[12], self.landmarks_array[1].landmark[24], self.landmarks_array[1].landmark[26])

    if (self.compare_angle(specBodyPart["sideProLeft"], specBodyPart["sideUserLeft"]) == 1):
        print("The left side of your body is straighter than desired. You should drop your left shoulder and lean in further by " + str(round(math.degrees(specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]))) + " degrees.")
    elif (self.compare_angle(specBodyPart["sideProLeft"], specBodyPart["sideUserLeft"]) == -1):
        print("Your left side of your body is more angled than desired. You should raise your left shoulder and straighten out your spine a bit more by " + str(round(math.degrees(specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]))) + " degrees.")
    # else:
        # print("Your left oblique side bend seems great! Keep up the good work.")

    if (self.compare_angle(specBodyPart["sideProRight"], specBodyPart["sideUserRight"]) == 1):
        print("The right side of your body is straighter than desired. You should drop your right shoulder and lean in further by " + str(round(math.degrees(specBodyPart["sideProRight"] - specBodyPart["sideUserRight"]))) + " degrees more.")
    elif (self.compare_angle(specBodyPart["sideProRight"],specBodyPart["sideUserRight"]) == -1):
        print("Your right side of your body is more angled than desired. You should raise your right shoulder and straighten out your spine a bit more by " + str(round(math.degrees(specBodyPart["sideProRight"] - specBodyPart["sideUserLeft"]))) + " degrees.")
    # else:
        # print("Your right oblique side bend seems great! Keep up the good work.")


  def get_2D_vector(self, a, b): 
    x = b.x - a.x
    y = b.y - a.y
    return [x, y]

    #todo: ADD BACK AXIS ROTATION
  def get_2D_angle(self, landmark_one, landmark_two, landmark_three): 
    # axis_dropped is the value that will not be considered in the coordinate point 
    # e.g if z is dropped, then the angle between two points in the x-y plane 
    vector_one = self.get_2D_vector(landmark_two, landmark_one)
    vector_two = self.get_2D_vector(landmark_two, landmark_three)
    cross_prod = vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1] 
    magnitude = math.sqrt(math.pow(vector_one[0], 2) + math.pow(vector_one[1], 2)) * math.sqrt(math.pow(vector_two[0], 2) + math.pow(vector_two[1], 2))
    # print (vector_one[0])
    # print (vector_one[1])
    # print (vector_two[0])
    # print (vector_two[1])
    angle = math.acos(cross_prod / magnitude)
    return angle
  
  def rotate(self): 
    # find the value for the amount that the user's landmarks need to rotate in order to be in the same orientation to the pros
    # Compare the two coordinates in 3d space, and then find the transformation matrix that transform the user’s image to the professional’s
    # Apply transformation matrix on every coordinate of user
    # rotate about the left hip 

    # find the angle between the two hips with the nose at the origins 
    nose = self.landmarks_array[1].landmark[0]
    user_left_hip = self.landmarks_array[0].landmark[23]
    print("user left hip", user_left_hip)
    pro_left_hip = self.landmarks_array[1].landmark[23]
    z_angle = self.get_2D_angle("z",user_left_hip, nose, pro_left_hip)  # angle in the x-y plane- angle of rotation about z axis 
    y_angle = self.get_2D_angle("y",user_left_hip, nose, pro_left_hip)
    x_angle = self.get_2D_angle("x",user_left_hip, nose, pro_left_hip)
    print(z_angle)
    print(y_angle) 
    print(x_angle)

    x_rotation_matrix = np.array([[1,0,0],
                        [0,math.cos(x_angle), -math.sin(x_angle)],
                        [0,math.sin(x_angle), math.cos(x_angle)]])
    y_rotation_matrix = np.array([[math.cos(y_angle), 0 ,math.sin(y_angle)],
                        [0, 1, 0],
                        [-math.sin(y_angle),0, math.cos(y_angle)]])
    z_rotation_matrix = np.array([[math.cos(z_angle),-math.sin(z_angle),0],
                        [math.cos(z_angle), -math.sin(z_angle),0],
                        [0,0,1]])
    
  
    for point in self.landmarks_array[0].landmark: 
      point_array = np.array([point.x, point.y, point.z])
      point_array = np.dot(x_rotation_matrix,point_array)
      point_array = np.dot(y_rotation_matrix,point_array)
      point_array = np.dot(y_rotation_matrix,point_array)
      point.x = point_array[0]
      point.y = point_array[1]
      point.z = point_array[2]
      #print(point.x, point.y, point.z)
        
  def get_angle(self, landmark_one, landmark_two, landmark_three):
    vector_one = self.get_vector(landmark_two, landmark_one)
    vector_two = self.get_vector(landmark_two, landmark_three)
    cross_prod = vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1] + vector_one[2] * vector_two[2]
    magnitude = math.sqrt(math.pow(vector_one[0], 2) + math.pow(vector_one[1], 2) + math.pow(vector_one[2],2)) * math.sqrt(math.pow(vector_two[0],2) + math.pow(vector_two[1],2) + math.pow(vector_two[2],2))
    angle = math.acos(cross_prod / magnitude)
    return angle

  def get_vector(self, a, b):
    x = b.x - a.x
    y = b.y - a.y
    z = b.z - a.z
    coordinate_list = [x, y, z]
    return coordinate_list
  def test(self):
        self.landmarks_array[0].landmark[13].x = 0 #test
        self.landmarks_array[0].landmark[13].y = 0 #test
        self.landmarks_array[0].landmark[13].z = 0 #test

        self.landmarks_array[0].landmark[11].x = 0 #test
        self.landmarks_array[0].landmark[11].y = 1 #test
        self.landmarks_array[0].landmark[11].z = 0 #test

        self.landmarks_array[0].landmark[15].x = 1 #test
        self.landmarks_array[0].landmark[15].y = 0 #test
        self.landmarks_array[0].landmark[15].z = 0 #test
        print(self.get_2D_angle(self.landmarks_array[0].landmark[11], self.landmarks_array[0].landmark[13], self.landmarks_array[0].landmark[15]))

pd = pose_detection("./test_inputs/charlie2_user.jpg", "./test_inputs/charlie2_pro.jpg")
pd.detect_pose()
pd.transform()
pd.scale()
pd.body_check()
# pd.show(0)
# pd.show(1)
pd.double_show()
# pd.test()

