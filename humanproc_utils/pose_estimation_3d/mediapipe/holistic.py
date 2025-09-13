
import cv2
import collections
import mediapipe as mp

from .visualize import draw_poses
from .filtering import filter_unseen_parts
from .extract_landmarks import get_landmarks
from .smoothing import moving_average_filter_dict

from .constants import FULL_FACEMESH_LANDMARKS_NAMES_LIST
from .constants import HAND_CONNECTIONS, HAND_LANDMARKS_NAMES
from .constants import POSE_CONNECTIONS, POSE_LANDMARKS_NAMES


#------------------------------------------------------------------------------------------------------
def extract_3d_human_keypoints_holistic(config):
  print(" ================ Keypoints Extraction ================ ")
  extraction_type = config.keypoints_type
  pose_data, face_data = collections.OrderedDict(), collections.OrderedDict()
  left_hand_data, right_hand_data = collections.OrderedDict(), collections.OrderedDict()

  mp_holistic = mp.solutions.holistic.Holistic( static_image_mode = False,
                                                model_complexity = config.model_complexity,
                                                smooth_landmarks = True,
                                                enable_segmentation = False,
                                                refine_face_landmarks = config.refine_face_landmarks,
                                                min_detection_confidence = config.min_detection_confidence, 
                                                min_tracking_confidence = config.min_tracking_confidence
                                            )

  cap = cv2.VideoCapture(config.source)
  frame_count = 0
  while cap.isOpened():

    success, image = cap.read()
    if not success:
      break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_holistic.process(image)

    is_pose_detected, is_left_hand_detected, is_right_hand_detected, is_face_detected = False, False, False, False
    #----------------------------------------------------------------------------------------------
    if results.pose_landmarks and "poses" in extraction_type:
      pose_data[str(frame_count)] = get_landmarks(results.pose_landmarks, landmarks_names=POSE_LANDMARKS_NAMES)
      is_pose_detected = True

    #----------------------------------------------------------------------------------------------
    if results.left_hand_landmarks and "hands" in extraction_type:
      left_hand_data[str(frame_count)] = get_landmarks(results.left_hand_landmarks, landmarks_names=HAND_LANDMARKS_NAMES)
      is_left_hand_detected = True

    #----------------------------------------------------------------------------------------------
    if results.right_hand_landmarks and "hands" in extraction_type:
      right_hand_data[str(frame_count)] = get_landmarks(results.right_hand_landmarks, landmarks_names=HAND_LANDMARKS_NAMES)
      is_right_hand_detected = True

    #----------------------------------------------------------------------------------------------
    if results.face_landmarks and "face" in extraction_type:
      face_data[str(frame_count)] = get_landmarks(results.face_landmarks, landmarks_names=FULL_FACEMESH_LANDMARKS_NAMES_LIST)
      is_face_detected = True

    #----------------------------------------------------------------------------------------------
    # Display the image with landmarks
    if config.show_keypoints:
      image.flags.writeable = True
      if is_pose_detected and "poses" in extraction_type: image = draw_poses(pose_data[str(frame_count)], image, connections=POSE_CONNECTIONS, landmarks_names=POSE_LANDMARKS_NAMES, name="Pose", text_position=(10, 25))
      if is_left_hand_detected and "hands" in extraction_type: image = draw_poses(left_hand_data[str(frame_count)], image, connections=HAND_CONNECTIONS, landmarks_names=HAND_LANDMARKS_NAMES, LINE_COLOR=(255, 255, 0), name="Left Hand", text_position=(10, 65))
      if is_right_hand_detected and "hands" in extraction_type: image = draw_poses(right_hand_data[str(frame_count)], image, connections=HAND_CONNECTIONS, landmarks_names=HAND_LANDMARKS_NAMES, LINE_COLOR=(0, 0, 255), name="Right Hand", text_position=(10, 85))
      if is_face_detected and "face" in extraction_type: image = draw_poses(face_data[str(frame_count)], image, connections=None, landmarks_names=FULL_FACEMESH_LANDMARKS_NAMES_LIST, name="Face", text_position=(10, 45), LANDMARK_RADIUS=1)

      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

    frame_count += 1

  cap.release()
  cv2.destroyAllWindows()

  mp_holistic.close()
  del mp_holistic

  if config.is_smoothing and "poses" in extraction_type:
    pose_data = moving_average_filter_dict(pose_data, window_size=5)
  if config.is_filtering_missing_parts and "poses" in extraction_type:
    pose_data = filter_unseen_parts(pose_data, offset=0.2)

  return {"poses":pose_data, "left_hand":left_hand_data, "right_hand":right_hand_data, "face":face_data}