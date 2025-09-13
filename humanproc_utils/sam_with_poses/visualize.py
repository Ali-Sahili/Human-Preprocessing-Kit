


import matplotlib.pyplot as plt

from .utils import get_min_max


#-----------------------------------------------------------------------------------------
def plot_segments(sam_results):
  fig = plt.figure()
  ax = fig.add_subplot(111)#, projection='3d')

  for person_id in sam_results:
    out_segs = sam_results[person_id]["segments_n"]
    nb_frames = len(out_segs)
    for frame_idx, segments_per_frame in enumerate(out_segs):
      ax.cla()  # Clear plot for each new frame

      ax.scatter(segments_per_frame[:,0], -segments_per_frame[:,1], c="red")

      ax.set_xlim([0., 1])
      ax.set_ylim([-1, 0])
      ax.set_xlabel("X")
      ax.set_ylabel("Y")
      ax.set_title(f"2D Poses - Frame {int(frame_idx) + 1} / {nb_frames}")
      plt.pause(0.25) 

    plt.show()

#-----------------------------------------------------------------------------------------
def plot_2d_data(segments, joints2d, inside_points, marging = 25):
  nb_frames = len(segments)
  min_values, max_values = get_min_max(segments)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  for frame_idx in range(len(segments)):
    ax.cla()
    ax.scatter(inside_points[frame_idx][:, 0], -inside_points[frame_idx][:, 1], color='blue')
    ax.plot(segments[frame_idx][:, 0], -segments[frame_idx][:, 1], 'r--', linewidth=2)
    ax.scatter(joints2d[frame_idx][:, 1], -joints2d[frame_idx][:, 0], color='purple', s=40)

    ax.set_xlim([min_values[0]-marging, max_values[0]+marging])
    ax.set_ylim([-max_values[1]-marging, -min_values[1]+marging])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Masks and Poses - Frame {int(frame_idx) + 1} / {nb_frames}")
    plt.pause(0.05)

  plt.show()