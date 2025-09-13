
from .hands_utils import extract_hand_keypoints
from .constants import JOINTS_INDICES, RIGHT_HAND_FINGERS, LEFT_HAND_FINGERS

#---------------------------------------------------------------------------------
def prepare_keypoints(joints3d, verts):

  keypoints = []
  for i in range(0, joints3d.shape[0]):

    joints_per_frame = {}
    for j_name, j_idx in JOINTS_INDICES.items():
      joints_per_frame[j_name] = joints3d[i][int(j_idx)]

    # RIGHT HAND
    right_hand_keypoints = extract_hand_keypoints(verts, landmark_indices=RIGHT_HAND_FINGERS)
    right_hand_coords = right_hand_keypoints[i]
    # right_hand_coords['WRIST'] = joints3d[i][4]

    # LEFT HAND
    left_hand_keypoints = extract_hand_keypoints(verts, landmark_indices=LEFT_HAND_FINGERS)
    left_hand_coords = left_hand_keypoints[i]
    # left_hand_coords['WRIST'] = joints3d[i][7]

    right_hand_coords = dict(("R_{}".format(k),v) for k,v in right_hand_coords.items())
    left_hand_coords = dict(("L_{}".format(k),v) for k,v in left_hand_coords.items())

    keypoints.append({**joints_per_frame, **right_hand_coords, **left_hand_coords})
  return keypoints