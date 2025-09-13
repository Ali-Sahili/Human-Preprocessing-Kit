
import numpy as np
from scipy.signal import savgol_filter  # For smoothing

from .utils import dict_to_lists_of_coords
from .transform import convert_to_relative_poses
from .constants import POSE_LANDMARKS_NAMES


#----------------------------------------------------------------------------------------------
def prepare_keypoints(keypoints_org, scale_factor = 0.5, is_smooth=True):
  print(" ================= Prepare Keypoints ================ ")
  keypoints = convert_to_relative_poses(keypoints_org)
  pose_3d_sequence, pose_frames_inds = dict_to_lists_of_coords(keypoints["poses"])
  
  if is_smooth:
    pose_3d_sequence = savgol_filter(pose_3d_sequence, window_length=5, polyorder=2, axis=0)

  left_hand_3d_sequence, lh_frames_inds = dict_to_lists_of_coords(keypoints["left_hand"])
  right_hand_3d_sequence, rh_frames_inds = dict_to_lists_of_coords(keypoints["right_hand"])
  face_mesh_3d_sequence, fm_frames_inds = dict_to_lists_of_coords(keypoints["face"])

  pose_3d_sequence_ready = np.empty_like(pose_3d_sequence)
  left_hand_3d_sequence_ready, right_hand_3d_sequence_ready, face_mesh_3d_sequence_ready = [], [], []

  lh_idx, rh_idx, fm_idx = 0, 0, 0
  for idx in range(pose_3d_sequence.shape[0]):
    left_wrist = pose_3d_sequence[idx, POSE_LANDMARKS_NAMES.index("LEFT_WRIST")]
    right_wrist = pose_3d_sequence[idx, POSE_LANDMARKS_NAMES.index("RIGHT_WRIST")]
    nose = pose_3d_sequence[idx, POSE_LANDMARKS_NAMES.index("NOSE")]

    #-----------------------------------------------------------------------------------------
    x_vals = pose_3d_sequence[idx,:,0]
    y_vals = pose_3d_sequence[idx,:,2]
    z_vals = -pose_3d_sequence[idx,:,1]
    pose_3d_sequence_ready[idx] = np.vstack((x_vals, y_vals, z_vals)).T

    #-----------------------------------------------------------------------------------------
    if idx in lh_frames_inds:
      x_vals_lh = (left_hand_3d_sequence[lh_idx,:,0] - left_hand_3d_sequence[lh_idx,0,0]) * scale_factor + left_wrist[0]
      y_vals_lh = (left_hand_3d_sequence[lh_idx,:,2] - left_hand_3d_sequence[lh_idx,0,2]) * scale_factor + left_wrist[2]
      z_vals_lh = -(left_hand_3d_sequence[lh_idx,:,1]- left_hand_3d_sequence[lh_idx,0,1]) * scale_factor - left_wrist[1]
      lh_idx += 1
      left_hand_3d_sequence_ready.append(np.vstack((x_vals_lh, y_vals_lh, z_vals_lh)).T)
    
    #-----------------------------------------------------------------------------------------
    if idx in rh_frames_inds:
      x_vals_rh = (right_hand_3d_sequence[rh_idx,:,0] - right_hand_3d_sequence[rh_idx,0,0]) * scale_factor + right_wrist[0]
      y_vals_rh = (right_hand_3d_sequence[rh_idx,:,2] - right_hand_3d_sequence[rh_idx,0,2]) * scale_factor + right_wrist[2]
      z_vals_rh = -(right_hand_3d_sequence[rh_idx,:,1]- right_hand_3d_sequence[rh_idx,0,1]) * scale_factor - right_wrist[1]
      rh_idx += 1
      right_hand_3d_sequence_ready.append(np.vstack((x_vals_rh, y_vals_rh, z_vals_rh)).T)

    #-----------------------------------------------------------------------------------------
    if idx in fm_frames_inds:
      x_vals_face = (face_mesh_3d_sequence[fm_idx,:,0] - face_mesh_3d_sequence[fm_idx,1,0]) + nose[0]
      y_vals_face = (face_mesh_3d_sequence[fm_idx,:,2] - face_mesh_3d_sequence[fm_idx,1,2]) + nose[2]
      z_vals_face = -(face_mesh_3d_sequence[fm_idx,:,1]- face_mesh_3d_sequence[fm_idx,1,1]) - nose[1]
      fm_idx += 1
      face_mesh_3d_sequence_ready.append(np.vstack((x_vals_face, y_vals_face, z_vals_face)).T)

  return (  pose_3d_sequence_ready, 
            np.asarray(left_hand_3d_sequence_ready, dtype=np.float32), 
            np.asarray(right_hand_3d_sequence_ready, dtype=np.float32), 
            np.asarray(face_mesh_3d_sequence_ready, dtype=np.float32), 
            pose_frames_inds, lh_frames_inds, rh_frames_inds, fm_frames_inds
        )