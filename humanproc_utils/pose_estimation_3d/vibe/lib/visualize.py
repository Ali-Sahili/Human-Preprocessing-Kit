
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .constants import SKELETON_EDGES, RIGHT_HAND_FINGERS, LEFT_HAND_FINGERS
from .hands_utils import extract_hand_keypoints, plot_fingers

#---------------------------------------------------------------------------------
def plot_3d_pose_sequence(joints3d, verts, add_mesh=True, add_fingers = True, frame_step=5,
                          remove_axis=False, remove_axis_labels=False, nb_frames_to_visualize=0):
  """
  Visualize the 3D pose sequence and vertices from VIBE output.

  Parameters:
  - joints3d: numpy array, shape (frames, joints, 3) representing 3D joints positions.
  - verts: numpy array, shape (frames, vertices, 3) representing 3D vertices positions.
  - frame_step: int, step for sampling frames for visualization.
  """
  num_frames, num_joints, _ = joints3d.shape

  # Create figure and 3D axis
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=-90, azim=-90)
  # # ax.view_init(elev=-10, azim=-133)
  # ax.view_init(elev=21, azim=-100)

  if nb_frames_to_visualize > 0 and nb_frames_to_visualize <= num_frames: 
    num_frames = nb_frames_to_visualize

  # Loop through selected frames and plot 3D skeleton and vertices
  for i in range(0, num_frames, frame_step):
    ax.cla()
    ax.set_title(f"Frame {i + 1}/{num_frames}")

    # Plot the skeleton joints
    joint_coords = joints3d[i]
    ax.scatter(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2], color='blue', s=10, label='Joints')

    # Plot connections between joints
    for start, end in SKELETON_EDGES:
      ax.plot(
          [joint_coords[start, 0], joint_coords[end, 0]],
          [joint_coords[start, 1], joint_coords[end, 1]],
          [joint_coords[start, 2], joint_coords[end, 2]],
          color="blue", linewidth=2
      )

    if add_fingers:
      # RIGHT HAND
      right_hand_keypoints = extract_hand_keypoints(verts, landmark_indices=RIGHT_HAND_FINGERS)
      right_hand_coords = right_hand_keypoints[i]
      r_hand_coords_np = np.array(list(right_hand_coords.values()))
      ax.scatter(r_hand_coords_np[:, 0], r_hand_coords_np[:, 1], r_hand_coords_np[:, 2], color='green', s=10, label='Joints')
      right_hand_coords['WRIST'] = joint_coords[4]
      plot_fingers(ax, right_hand_coords)

      # LEFT HAND
      left_hand_keypoints = extract_hand_keypoints(verts, landmark_indices=LEFT_HAND_FINGERS)
      left_hand_coords = left_hand_keypoints[i]
      l_hand_coords_np = np.array(list(left_hand_coords.values()))
      ax.scatter(l_hand_coords_np[:, 0], l_hand_coords_np[:, 1], l_hand_coords_np[:, 2], color='green', s=10, label='Joints')
      left_hand_coords['WRIST'] = joint_coords[7]
      plot_fingers(ax, left_hand_coords)

    # Plot the vertices as a mesh
    if add_mesh:
      verts_coords = verts[i]
      ax.scatter(verts_coords[:, 0], verts_coords[:, 1], verts_coords[:, 2], color='orange', s=1, alpha=0.5, label='Vertices')

    # Set limits and labels for better visibility
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if remove_axis: ax.set_axis_off()
    if remove_axis_labels: ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    plt.pause(0.1)
  
  plt.show()
