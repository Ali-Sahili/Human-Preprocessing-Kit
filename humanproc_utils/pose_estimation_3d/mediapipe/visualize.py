
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils import get_min_max_per_keypoints, is_empty_numpy
from .constants import POSE_CONNECTIONS, POSE_LANDMARKS_NAMES
from .constants import HAND_CONNECTIONS, HAND_LANDMARKS_NAMES
from .constants import REDUNDANCY_FACE_MESH_TO_REMOVE
from .constants import REDUNDANCY_LEFT_HAND_TO_REMOVE, REDUNDANCY_RIGHT_HAND_TO_REMOVE

#------------------------------------------------------------------------------------------------------
# Convert landmarks to list of (x, y) coordinates in image pixels
def landmarks_to_pixels(landmarks, image_width, image_height, landmarks_names):
  if isinstance(landmarks_names, dict):
    landmarks_names = list(landmarks_names.keys())
  return [(int(landmarks[landmark].get('x', 0) * image_width),
           int(landmarks[landmark].get('y', 0) * image_height))
           for landmark in landmarks_names]

#------------------------------------------------------------------------------------------------------
def draw_poses(landmarks, image, connections, landmarks_names, name="poses", text_position=(10, 75),
      LANDMARK_COLOR=(0, 255, 0), LINE_COLOR=(255, 0, 0), LANDMARK_RADIUS = 5, LINE_THICKNESS = 2):
  
  image_height, image_width, _ = image.shape

  # Get normalized landmarks for drawing
  normalized_landmarks = landmarks_to_pixels(landmarks, image_width, image_height, landmarks_names)

  # Draw landmarks
  for point in normalized_landmarks:
    cv2.circle(image, point, LANDMARK_RADIUS, LANDMARK_COLOR, -1)

  # Draw connections based on POSE_CONNECTIONS
  if connections:
    for start_idx, end_idx in connections:
      cv2.line(image, normalized_landmarks[start_idx], normalized_landmarks[end_idx], LINE_COLOR, LINE_THICKNESS)

  cv2.putText(image, f"{name} detected", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, LANDMARK_COLOR, 2)

  return image

#------------------------------------------------------------------------------------------------------
def plot_3d_skeleton_sequence(keypoints, add_hands = False, scale_factor=.5):
  """ Plots a 3D skeleton sequence with body and hand keypoints over time, aligned in depth. """
  print(" ================= Plot 3d Skeleton ================= ")
  if not isinstance(keypoints, dict):
    print("Keypoints are not in the right FORMAT !")
    return
    
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  min_x, max_x, min_y, max_y, min_z, max_z = get_min_max_per_keypoints(keypoints)
  pose_data, left_hand_data, right_hand_data = keypoints["poses"], keypoints["left_hand"], keypoints["right_hand"]
  nb_frames = len(list(pose_data.keys()))
  
  for frame_idx, pose_frame in pose_data.items():
    ax.cla()  # Clear plot for each new frame

    #-----------------------------------------------------------------------------------------
    left_wrist = pose_frame.get("LEFT_WRIST", {"x": 0, "y": 0, "z": 0})
    right_wrist = pose_frame.get("RIGHT_WRIST", {"x": 0, "y": 0, "z": 0})

    #-----------------------------------------------------------------------------------------
    x_vals = [pose_frame[l_name]['x'] for l_name in POSE_LANDMARKS_NAMES]
    y_vals = [pose_frame[l_name]['z'] for l_name in POSE_LANDMARKS_NAMES]
    z_vals = [-pose_frame[l_name]['y'] for l_name in POSE_LANDMARKS_NAMES]

    # Plot each connection in the skeleton
    for start, end in POSE_CONNECTIONS:
      ax.plot(
        [x_vals[start], x_vals[end]],
        [y_vals[start], y_vals[end]],
        [z_vals[start], z_vals[end]],
        'blue', #linewidth=2, markersize=4
      )

    #-----------------------------------------------------------------------------------------
    if add_hands:
      left_hand_frame = left_hand_data.get(frame_idx)
      if left_hand_frame:
        x_vals_lh = [left_hand_frame[l_name]['x'] for l_name in HAND_LANDMARKS_NAMES]
        y_vals_lh = [left_hand_frame[l_name]['z'] for l_name in HAND_LANDMARKS_NAMES]
        z_vals_lh = [-left_hand_frame[l_name]['y'] for l_name in HAND_LANDMARKS_NAMES]

        # Plot each connection in the skeleton
        for start, end in HAND_CONNECTIONS:
          ax.plot(
            [(x_vals_lh[start]-x_vals_lh[0]) * scale_factor + left_wrist['x'], (x_vals_lh[end]-x_vals_lh[0]) * scale_factor + left_wrist['x']],
            [(y_vals_lh[start]-y_vals_lh[0]) * scale_factor + left_wrist['z'], (y_vals_lh[end]-y_vals_lh[0]) * scale_factor + left_wrist['z']],
            [(z_vals_lh[start]-z_vals_lh[0]) * scale_factor - left_wrist['y'], (z_vals_lh[end]-z_vals_lh[0]) * scale_factor - left_wrist['y']],
            'red' #, linewidth=1.5, markersize=3
          )

      #-----------------------------------------------------------------------------------------
      right_hand_frame = right_hand_data.get(frame_idx)
      if right_hand_frame:
        x_vals_rh = [right_hand_frame[l_name]['x'] for l_name in HAND_LANDMARKS_NAMES]
        y_vals_rh = [right_hand_frame[l_name]['z'] for l_name in HAND_LANDMARKS_NAMES]
        z_vals_rh = [-right_hand_frame[l_name]['y'] for l_name in HAND_LANDMARKS_NAMES]

        # Plot each connection in the skeleton
        for start, end in HAND_CONNECTIONS:
          ax.plot(
            [(x_vals_rh[start]-x_vals_rh[0]) * scale_factor + right_wrist['x'], (x_vals_rh[end]-x_vals_rh[0]) * scale_factor + right_wrist['x']],
            [(y_vals_rh[start]-y_vals_rh[0]) * scale_factor + right_wrist['z'], (y_vals_rh[end]-y_vals_rh[0]) * scale_factor + right_wrist['z']],
            [(z_vals_rh[start]-z_vals_rh[0]) * scale_factor - right_wrist['y'], (z_vals_rh[end]-z_vals_rh[0]) * scale_factor - right_wrist['y']],
            'red' #, linewidth=1.5, markersize=3
          )

    #-----------------------------------------------------------------------------------------
    # Set plot limits and labels
    ax.set_xlim([min_x-min_x/5, max_x+max_x/5])
    ax.set_ylim([min_z-min_z/5, max_z+max_z/5])
    ax.set_zlim([-(max_y+max_y/5), -(min_y-min_y/5)])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Skeleton - Frame {int(frame_idx) + 1} / {nb_frames}")
    plt.pause(0.05)  # Adjust pause time for animation speed

  plt.show()

#------------------------------------------------------------------------------------------------------
def visualize_3d_skeleton_numpy(pose_3d_sequence_ready, left_hand_3d_sequence_ready, right_hand_3d_sequence_ready, 
                            face_mesh_3d_sequence_ready, lh_frames_inds, rh_frames_inds, fm_frames_inds):  
  print(" ================= Final Plotting =================== ")
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=10., azim=-80)

  no_left_hand = is_empty_numpy(left_hand_3d_sequence_ready)
  no_right_hand = is_empty_numpy(right_hand_3d_sequence_ready)
  no_face_mesh = is_empty_numpy(face_mesh_3d_sequence_ready)

  lh_idx, rh_idx, fm_idx = 0, 0, 0
  nb_frames = pose_3d_sequence_ready.shape[0]
  for idx in range(nb_frames):
    ax.cla()

    #-----------------------------------------------------------------------------------------
    for start, end in POSE_CONNECTIONS:
      if not no_left_hand:
        if start in REDUNDANCY_LEFT_HAND_TO_REMOVE or end in REDUNDANCY_LEFT_HAND_TO_REMOVE:
          continue
      if not no_right_hand:
        if start in REDUNDANCY_RIGHT_HAND_TO_REMOVE or end in REDUNDANCY_RIGHT_HAND_TO_REMOVE:
          continue
      if not no_face_mesh:
        if start in REDUNDANCY_FACE_MESH_TO_REMOVE or end in REDUNDANCY_FACE_MESH_TO_REMOVE:
          continue

      ax.plot(
        [pose_3d_sequence_ready[idx,start,0], pose_3d_sequence_ready[idx,end,0]],
        [pose_3d_sequence_ready[idx,start,1], pose_3d_sequence_ready[idx,end,1]],
        [pose_3d_sequence_ready[idx,start,2], pose_3d_sequence_ready[idx,end,2]],
        'blue'
      )

    #-----------------------------------------------------------------------------------------
    if idx in lh_frames_inds:
      for start, end in HAND_CONNECTIONS:
        ax.plot(
          [left_hand_3d_sequence_ready[lh_idx,start,0], left_hand_3d_sequence_ready[lh_idx,end,0]],
          [left_hand_3d_sequence_ready[lh_idx,start,1], left_hand_3d_sequence_ready[lh_idx,end,1]],
          [left_hand_3d_sequence_ready[lh_idx,start,2], left_hand_3d_sequence_ready[lh_idx,end,2]],
          'red'
        )
      lh_idx += 1

    #-----------------------------------------------------------------------------------------
    if idx in rh_frames_inds:
      for start, end in HAND_CONNECTIONS:
        ax.plot(
          [right_hand_3d_sequence_ready[rh_idx,start,0], right_hand_3d_sequence_ready[rh_idx,end,0]],
          [right_hand_3d_sequence_ready[rh_idx,start,1], right_hand_3d_sequence_ready[rh_idx,end,1]],
          [right_hand_3d_sequence_ready[rh_idx,start,2], right_hand_3d_sequence_ready[rh_idx,end,2]],
          'red'
        )
      rh_idx += 1

    #-----------------------------------------------------------------------------------------
    if idx in fm_frames_inds:
      ax.scatter( face_mesh_3d_sequence_ready[fm_idx,:,0], 
                  face_mesh_3d_sequence_ready[fm_idx,:,1], 
                  face_mesh_3d_sequence_ready[fm_idx,:,2], 
                  color='green', s=1
                )
      fm_idx += 1

    #-----------------------------------------------------------------------------------------
    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_zlim([-1., 1.])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Poses - Frame {idx+1}/{nb_frames}")
    plt.pause(0.05)  # Adjust pause time for animation speed
  plt.show()