
import cv2
import collections
import mediapipe as mp

from .constants import *
from .visualize import draw_poses
from .filtering import filter_unseen_parts
from .smoothing import moving_average_filter_dict

#------------------------------------------------------------------------------------------------------
# 
def get_landmarks(mediapipe_landmarks, landmarks_names):
  frame_landmarks = {}
  for idx, landmark in enumerate(mediapipe_landmarks.landmark):
    frame_landmarks[landmarks_names[idx]] = {
        'x': landmark.x,
        'y': landmark.y,
        'z': landmark.z,
        'v': landmark.visibility
    }
  return frame_landmarks

#------------------------------------------------------------------------------------------------------
# 
def get_face_landmarks(mediapipe_landmarks, landmarks_names):
  frame_landmarks = {}
  for name, idx in landmarks_names.items():
    landmark = mediapipe_landmarks[0].landmark[idx]
    frame_landmarks[name] = {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        'v': landmark.visibility
    }
  return frame_landmarks

#------------------------------------------------------------------------------------------------------
# 
def extract_3d_human_keypoints(config):
  print(" ================ Keypoints Extraction ================ ")
  extraction_type = config.keypoints_type
  show_keypoints = config.show_keypoints

  pose_data, face_data = collections.OrderedDict(), collections.OrderedDict()
  left_hand_data, right_hand_data = collections.OrderedDict(), collections.OrderedDict()

  #------------------------------------------------------------------------------------------------------
  # Initialize MediaPipe model
  if "pose" in extraction_type:
    # mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_pose = mp.solutions.pose.Pose( static_image_mode = False,    # for video
                                      model_complexity = config.model_complexity,  # [0, 1, 2]
                                      smooth_landmarks = True,     # to reduce jitter
                                      enable_segmentation = False, # enable to generate segmentation mask
                                      min_detection_confidence = config.min_detection_confidence,
                                      min_tracking_confidence = config.min_tracking_confidence
                                    )

  if "hands" in extraction_type:
    mp_hands = mp.solutions.hands.Hands( static_image_mode = False, 
                                         max_num_hands = 2, 
                                         model_complexity = 1, # fix to one to prevent some errors
                                         min_detection_confidence = config.min_detection_confidence,
                                         min_tracking_confidence = config.min_tracking_confidence
                                      )
  if "face" in extraction_type:
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh( static_image_mode = False, 
                                                    max_num_faces = config.max_num_faces,
                                                    refine_landmarks = config.refine_face_landmarks,
                                                    min_detection_confidence = config.min_detection_confidence_face,
                                                    min_tracking_confidence = config.min_tracking_confidence_face
                                      )
    
  #------------------------------------------------------------------------------------------------------
  # Capture video
  cap = cv2.VideoCapture(config.source)
  frame_count = 0
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #----------------------------------------------------------------------------------------------
    if "pose" in extraction_type: 
      results = mp_pose.process(image_rgb)
      
      if results.pose_landmarks:
        pose_data[str(frame_count)] = get_landmarks(results.pose_landmarks, landmarks_names=POSE_LANDMARKS_NAMES)

        if show_keypoints:
          image = draw_poses(pose_data[str(frame_count)], image, connections=POSE_CONNECTIONS, landmarks_names=POSE_LANDMARKS_NAMES, name="Pose", text_position=(10, 25))
    
    #----------------------------------------------------------------------------------------------
    if "hands" in extraction_type: 
      results = mp_hands.process(image_rgb)

      if results.multi_hand_landmarks and results.multi_handedness:
        is_left_detected, is_right_detected = False, False
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
          hand_label = handedness.classification[0].label   # Determine hand label ('Left' or 'Right')  
          if hand_label == "Right":
            left_hand_data[str(frame_count)] = get_landmarks(hand_landmarks, landmarks_names = HAND_LANDMARKS_NAMES)
            is_left_detected = True
          else:
            right_hand_data[str(frame_count)] = get_landmarks(hand_landmarks, landmarks_names = HAND_LANDMARKS_NAMES)
            is_right_detected = True

        if show_keypoints:
          if is_left_detected:
            image = draw_poses(left_hand_data[str(frame_count)], image, connections=HAND_CONNECTIONS, landmarks_names=HAND_LANDMARKS_NAMES, LINE_COLOR=(255, 0, 0), name="Left Hand", text_position=(10, 65))
          if is_right_detected:
            image = draw_poses(right_hand_data[str(frame_count)], image, connections=HAND_CONNECTIONS, landmarks_names=HAND_LANDMARKS_NAMES, LINE_COLOR=(0, 0, 255), name="Right Hand", text_position=(10, 85))
    
    #----------------------------------------------------------------------------------------------
    if "face" in extraction_type: 
      results = mp_face_mesh.process(image_rgb)
      if results.multi_face_landmarks:
        face_data[str(frame_count)] = get_face_landmarks(results.multi_face_landmarks, landmarks_names=FULL_FACEMESH_LANDMARKS_NAMES)
        image = draw_poses(face_data[str(frame_count)], image, connections=None, landmarks_names=FULL_FACEMESH_LANDMARKS_NAMES, name="Face", text_position=(10, 45), LANDMARK_RADIUS=1)

    frame_count += 1
    #----------------------------------------------------------------------------------------------
    # Display the image with landmarks
    if show_keypoints:
      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

  cap.release()
  cv2.destroyAllWindows()

  if config.is_smoothing and bool(pose_data):
    pose_data = moving_average_filter_dict(pose_data, window_size=5)
  if config.is_filtering_missing_parts and "poses" in extraction_type:
    pose_data = filter_unseen_parts(pose_data, offset=0.2)

  return {"poses":pose_data, "left_hand":left_hand_data, "right_hand":right_hand_data, "face":face_data}