
from .constants import HAND_EDGES

#---------------------------------------------------------------------------------
def extract_hand_keypoints(verts, landmark_indices):
  """
  Extracts 3D hand keypoints from vertices based on selected vertex indices.

  Parameters:
   - verts: np.ndarray, shape (frames, vertices, 3) 3D mesh vertices for each frame.

  Returns:
   - hand_keypoints: np.ndarray, shape (frames, 10, 3) - Fingers
  """
  hand_keypoints = []

  for frame_verts in verts:
    frame_keypoints = {}
    for name, index in landmark_indices.items():
        frame_keypoints[name] = frame_verts[index]
    hand_keypoints.append(frame_keypoints)
  return hand_keypoints


#---------------------------------------------------------------------------------
def plot_fingers(ax, hand_coords, color="red"):

  for start, end in HAND_EDGES:
    ax.plot(
        [hand_coords[start][0], hand_coords[end][0]],
        [hand_coords[start][1], hand_coords[end][1]],
        [hand_coords[start][2], hand_coords[end][2]],
        color=color, linewidth=2
    )