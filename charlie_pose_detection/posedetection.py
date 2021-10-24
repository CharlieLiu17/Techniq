import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# app = Flask(__name__) 

# def execute():
#     return pose_detection(jump1.jpg, jump2.jpg)

# @app.route('/send_data', methods = ['POST'])

# def get_userInput():
#     image = request.files['image']
#     if image.filename != '':
#         image.save(image.filename)
#     return detect_pose(image)

# @app.route('/')

# def index():
#     return render_template('index.html', x = execute())

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
    # mp_drawing.plot_landmarks(
    #     self.world_landmarks[idx], mp_pose.POSE_CONNECTIONS)

  def doubleShow(self, idx):
    image = cv2.imread('./tmp/annotated_image' + str(1) + '.png')
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # # bilateral filter to "results.segmentation_mask" with "image".
    # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = BG_COLOR
    # annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    print(self.landmarks_array[0])
    mp_drawing.draw_landmarks(
        image,
        self.landmarks_array[0],
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('./tmp/annotated_image' + str(1) + '.png', image)

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
      user[length[1]].x = user[length[0]].x + ratio * vector[0]
      user[length[1]].y = user[length[0]].y + ratio * vector[1]
      user[length[1]].z = user[length[0]].z + ratio * vector[2]
    #self.landmarks_array[0] = user
    
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
    transformCode = [0,0,0]
    transformCode[0] = self.landmarks_array[1].landmark[0].x - self.landmarks_array[0].landmark[0].x 
    transformCode[1] = self.landmarks_array[1].landmark[0].y - self.landmarks_array[0].landmark[0].y
    transformCode[2] = self.landmarks_array[1].landmark[0].z - self.landmarks_array[0].landmark[0].z

    for bodyPart in self.landmarks_array[0].landmark:
      bodyPart.x = bodyPart.x + transformCode[0]
      bodyPart.y = bodyPart.y + transformCode[1]
      bodyPart.z = bodyPart.z + transformCode[2]
  
  def bodyCheck(self):
        print("Please wait while the advice on your form is being generated!")
        #arm
        specBodyPart["armUserLeft"] = get_angle(self, self.landmarks_array.landmark[0][11], self.landmarks_array.landmark[0][13], self.landmarks_array.landmark[0][15])

        specBodyPart["armProLeft"] = get_angle(self, self.landmarks_array.landmark[1][11], self.landmarks_array.landmark[1][13], self.landmarks_array.landmark[1][15])

        specBodyPart["armUserRight"] = get_angle(self, self.landmarks_array.landmark[0][12], self.landmarks_array.landmark[0][14], self.landmarks_array.landmark[0][16])

        specBodyPart["armProRight"] = get_angle(self, self.landmarks_array.landmark[1][12], self.landmarks_array.landmark[1][14], self.landmarks_array.landmark[1][16])


        # left arm
        if (compare_angle(specBodyPart["armUserLeft"], specBodyPart["armProLeft"]) == -1):
            print("You should extend your left elbow out more around " + (specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]) + " degrees more.")
        elif (compare_angle(specBodyPart["armUserLeft"], specBodyPart["armProLeft"]) == 1):
            print("Your left elbow is extended too far. You should contract your left elbow more towards your body by " + (specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]) + " degrees.")
        else:
            print("Your left elbow position seems great! Keep up the good work.")

        # right arm
        if (compare_angle(specBodyPart["armProRight"], specBodyPart["armUserRight"]) == -1):
            print("You should extend your right elbow out more around " + (specBodyPart["armProRight"] - specBodyPart["armUserRight"]) + " degrees more.")
        elif (compare_angle(specBodyPart["armProRight"],specBodyPart["armUserRight"]) == 1):
            print("Your right elbow is extended too far. You should contract your right elbow more towards your body by " + (specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]) + " degrees.")
        else:
            print("Your right elbow position seems great! Keep up the good work.")
        

        #leg/knee
        specBodyPart["kneeUserLeft"] = get_angle(self, self.landmarks_array.landmark[0][23], self.landmarks_array.landmark[0][25], self.landmarks_array.landmark[0][27])
        specBodyPart["kneeProLeft"] = get_angle(self, self.landmarks_array.landmark[1][23], self.landmarks_array.landmark[1][25], self.landmarks_array.landmark[1][27])
        specBodyPart["kneeUserRight"] = get_angle(self, self.landmarks_array.landmark[0][24], self.landmarks_array.landmark[0][26], self.landmarks_array.landmark[0][28])
        specBodyPart["kneeProRight"] = get_angle(self, self.landmarks_array.landmark[1][24], self.landmarks_array.landmark[1][26], self.landmarks_array.landmark[1][28])
        if (compare_angle(specBodyPart["kneeProLeft"], specBodyPart["kneeUserLeft"]) == -1):
            print("You should extend your left knee out more around " + (specBodyPart["kneeProLeft"] - specBodyPart["kneeUserLeft"]) + " degrees more.")
        elif (compare_angle(specBodyPart["kneeProLeft"], specBodyPart["kneeUserLeft"]) == 1):
            print("Your left knee is extended too far. You should contract your left knee towards your body by " + (specBodyPart["kneeProLeft"] - specBodyPart["kneeUserLeft"]) + " degrees.")
        else:
            print("Your left knee position seems great! Keep up the good work.")
        
        if (compare_angle(specBodyPart["kneeProRight"], specBodyPart["kneeUserRight"]) == -1):
            print("You should extend your right knee out more around " + (specBodyPart["kneeProRight"] - specBodyPart["kneeUserRight"]) + " degrees more.")
        elif (compare_angle(specBodyPart["kneeProRight"], specBodyPart["kneeUserRight"]) == 1):
            print("Your right knee is extended too far. You should contract your right knee towards your body. Contract it closer to your body by " + (specBodyPart["armProLeft"] - specBodyPart["armUserLeft"]) + " degrees.")
        else:
            print("Your right knee position seems great! Keep up the good work.")
        

        #hip
        specBodyPart["hipUserLeft"] = get_angle(self, self.landmarks_array.landmark[0][24], self.landmarks_array.landmark[0][23], self.landmarks_array.landmark[0][25])
        specBodyPart["hipProLeft"] = get_angle(self, self.landmarks_array.landmark[1][24], self.landmarks_array.landmark[1][23], self.landmarks_array.landmark[1][25])
        specBodyPart["hipUserRight"] = get_angle(self, self.landmarks_array.landmark[0][23], self.landmarks_array.landmark[0][24], self.landmarks_array.landmark[0][26])
        specBodyPart["hipProRight"] = get_angle(self, self.landmarks_array.landmark[1][23], self.landmarks_array.landmark[1][24], self.landmarks_array.landmark[1][26])

        if (compare_angle(specBodyPart["hipProLeft"], specBodyPart["hipUserLeft"]) == 1):
            print("You should extend your left leg out more by around " + (specBodyPart["hipProLeft"] - specBodyPart["hipUserLeft"]) + " degrees more.")
        elif (compare_angle(specBodyPart["hipProLeft"], specBodyPart["hipUserLeft"] == -1)):
            print("Your left leg is extended too far outwards. You should bring in your left leg by " + (specBodyPart["hipProLeft"] - specBodyPart["hipUserLeft"]) + " degrees.")
        else:
            print("Your leg position seems great! Keep up the good work.")
        

        if (compare_angle(specBodyPart["hipProRight"], specBodyPart["hipUserRight"]) == 1):
            print("You should extend your right leg out more by around " + (specBodyPart["hipProRight"] - specBodyPart["hipUserRight"]) + " degrees more.")
        elif (compare_angle(specBodyPart["hipProRight"], specBodyPart["hipUserRight"]) == -1):
            print("Your right leg is extended too far outwards. You should bring your right leg in by " + (specBodyPart["hipProRight"] - specBodyPart["hipUserRight"]) + " degrees.")
        else:
            print("Your right leg position seems great! Keep up the good work.")


        #side body
        specBodyPart["sideUserLeft"] = get_angle(self, self.landmarks_array.landmark[0][11], self.landmarks_array.landmark[0][23], self.landmarks_array.landmark[0][25])
        specBodyPart["sideProLeft"] = get_angle(self, self.landmarks_array.landmark[1][11], self.landmarks_array.landmark[1][23], self.landmarks_array.landmark[1][25])
        specBodyPart["sideUserRight"] = get_angle(self, self.landmarks_array.landmark[0][12], self.landmarks_array.landmark[0][24], self.landmarks_array.landmark[0][26])
        specBodyPart["sideProRight"] = get_angle(self, self.landmarks_array.landmark[1][12], self.landmarks_array.landmark[1][24], self.landmarks_array.landmark[1][26])

        if (compare_angle(specBodyPart["sideProLeft"], specBodyPart["sideUserLeft"]) == 1):
            print("The left side of your body is straighter than desired. You should drop your left shoulder and lean in further by " + (specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]) + " degrees.")
        elif (compare_angle(specBodyPart["sideProLeft"], specBodyPart["sideUserLeft"]) == -1):
            print("Your left side of your body is more angled than desired. You should raise your left shoulder and straighten out your spine a bit more by " + (specBodyPart["sideProLeft"] - specBodyPart["sideUserLeft"]) + " degrees.")
        else:
            print("Your left oblique side bend seems great! Keep up the good work.")

        if (compare_angle(specBodyPart["sideProRight"], specBodyPart["sideUserRight"]) == 1):
            print("The right side of your body is straighter than desired. You should drop your right shoulder and lean in further by " + (specBodyPart["sideProRight"] - specBodyPart["sideUserRight"]) + " degrees more.")
        elif (compare_angle(specBodyPart["sideProRight"],specBodyPart["sideUserRight"]) == -1):
            print("Your right side of your body is more angled than desired. You should raise your right shoulder and straighten out your spine a bit more by " + (specBodyPart["sideProRight"] - specBodyPart["sideUserLeft"]) + " degrees.")
        else:
            print("Your right oblique side bend seems great! Keep up the good work.")

        #arm pit
        specBodyPart["armPitUserLeft"] = get_angle(self, self.landmarks_array.landmark[0][13], self.landmarks_array.landmark[0][11], self.landmarks_array.landmark[0][23])
        specBodyPart["armPitProLeft"] = get_angle(self, self.landmarks_array.landmark[1][13], self.landmarks_array.landmark[1][11], self.landmarks_array.landmark[1][23])
        specBodyPart["armPitUserRight"] = get_angle(self, self.landmarks_array.landmark[0][14], self.landmarks_array.landmark[0][12], self.landmarks_array.landmark[0][24])
        specBodyPart["armPitProRight"] = get_angle(self, self.landmarks_array.landmark[1][14], self.landmarks_array.landmark[1][12], self.landmarks_array.landmark[1][24])

        if (compare_angle(specBodyPart["armPitProLeft"], specBodyPart["armPitUserLeft"]) == 1):
            print("Your left arm is dropped to low to your side. Your left arm should be raised up more by " + (specBodyPart["armPitProLeft"] - specBodyPart["armPitUserLeft"]) + " degrees.")
        elif (compare_angle(specBodyPart["armPitProLeft"], specBodyPart["armPitUserLeft"]) == -1):
            print("Your left arm is raised to high. Your left arm should be lowered down to your side more by " + (specBodyPart["armPitProLeft"] - specBodyPart["armPitUserLeft"]) + " degrees.")
        else:
            print("Your left arm position looks great! Keep up the good work.")

        if (compare_angle(specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]) == 1):
            print("Your right arm is dropped to low to your side. Your right arm should be raised up more by " + (specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]) + " degrees more.")
        elif (compare_angle(specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]) == -1):
            print("Your right arm is raised to high. Your right arm should be lowered down to your side more by " + (specBodyPart["armPitProRight"] - specBodyPart["armPitUserRight"]) + " degrees.")
        else:
            print("Your right arm position looks great! Keep up the good work.")

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
    
    self.landmarks_array.landmark[0] *= x_rotation_matrix
    self.landmarks_array.landmark[0] *= y_rotation_matrix
    self.landmarks_array.landmark[0] *= z_rotation_matrix
        
  def get_angle(self, landmark_one, landmark_two, landmark_three):
    vector_one = get_vector(landmark_two, landmark_one)
    vector_two = get_vector(landmark_two, landmark_three)
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


pd = pose_detection("./jump1.jpg", "./jump2.jpg")
pd.detect_pose()
pd.show(0)
pd.transform()
#pd.scale()
pd.doubleShow(0)

# if __name__ == "__main__":
#        app.run(host='0.0.0.0', debug=True)
#def video_upload_1 () : 
    # upload the first video 