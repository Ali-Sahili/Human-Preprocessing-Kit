
from collections import OrderedDict

from .constants import ROOT_IN_BETWEEN


#------------------------------------------------------------------------------------------------------
def convert_to_relative_poses(keypoints):
  pose_data = keypoints["poses"]
  face_data = keypoints["face"]
  left_hand_data, right_hand_data = keypoints["left_hand"], keypoints["right_hand"]

  if not bool(pose_data):
    print("NO poses FOUND!")
    return None

  relative_pose_data = OrderedDict()
  relative_left_hand_data = OrderedDict()
  relative_right_hand_data = OrderedDict()
  relative_face_data = OrderedDict()
  for frame_idx, pose_frame in pose_data.items():
    
    left_hip, right_hip = pose_frame["LEFT_HIP"], pose_frame["RIGHT_HIP"]

    # Compute the root keypoint as the midpoint between left and right hip
    root_x = (left_hip['x'] + right_hip['x']) / 2
    root_y = (left_hip['y'] + right_hip['y']) / 2
    root_z = (left_hip['z'] + right_hip['z']) / 2

    # POSE
    r_pose_data = OrderedDict()
    for l_name, l_coords in pose_frame.items():
      r_pose_data[l_name] = OrderedDict( {
                                'x': l_coords['x'] - root_x,
                                'y': l_coords['y'] - root_y,
                                'z': l_coords['z'] - root_z,
                            })
    relative_pose_data[frame_idx] = r_pose_data
    
    # LEFT HAND
    left_hand_frame = left_hand_data.get(frame_idx)
    if left_hand_frame:
      r_lh_data = OrderedDict()
      for l_name, l_coords in left_hand_frame.items():
        r_lh_data[l_name] = OrderedDict( {
                                'x': l_coords['x'] - root_x,
                                'y': l_coords['y'] - root_y,
                                'z': l_coords['z'] - root_z,
                            })
      relative_left_hand_data[frame_idx] = r_lh_data
    
    # RIGHT HAND
    right_hand_frame = right_hand_data.get(frame_idx)
    if right_hand_frame:
      r_rh_data = OrderedDict()
      for l_name, l_coords in right_hand_frame.items():
        r_rh_data[l_name] = OrderedDict( {
                                'x': l_coords['x'] - root_x,
                                'y': l_coords['y'] - root_y,
                                'z': l_coords['z'] - root_z,
                            })
      relative_right_hand_data[frame_idx] = r_rh_data

    # FACE MESH
    face_data_frame = face_data.get(frame_idx)
    if face_data_frame:
      r_face_data = OrderedDict()
      for l_name, l_coords in face_data_frame.items():
        r_face_data[l_name] = OrderedDict( {
                                'x': l_coords['x'] - root_x,
                                'y': l_coords['y'] - root_y,
                                'z': l_coords['z'] - root_z,
                            })
      relative_face_data[frame_idx] = r_face_data

  return {"poses": relative_pose_data, "left_hand": relative_left_hand_data, 
            "right_hand": relative_right_hand_data, "face": relative_face_data}
