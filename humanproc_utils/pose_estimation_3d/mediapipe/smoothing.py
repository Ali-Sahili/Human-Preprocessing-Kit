
import numpy as np

#------------------------------------------------------------------------------------------------------
def moving_average_filter_dict(pose_data, window_size=5):
  print(" ================= Smoothing ======================== ")
  smoothed_data = {}

  frame_0 = int(list(pose_data.keys())[0])
  for frame_idx in pose_data.keys():    
    i = int(frame_idx) - frame_0
    smoothed_frame = {}
    for key in pose_data[frame_idx].keys():
      x_vals, y_vals, z_vals = [], [], []
      for j in range(max(0, i - window_size // 2), min(len(pose_data), i + window_size // 2 + 1)):
        j_idx = str(j+frame_0)
        if j_idx in pose_data.keys():
          x_vals.append(pose_data[j_idx][key]["x"])
          y_vals.append(pose_data[j_idx][key]["y"])
          z_vals.append(pose_data[j_idx][key]["z"])

      # Calculate the average for x, y, z within the window
      avg_x = np.mean(x_vals)
      avg_y = np.mean(y_vals)
      avg_z = np.mean(z_vals)
      
      # Update the smoothed frame with averaged values
      smoothed_frame[key] = {"x": avg_x, "y": avg_y, "z": avg_z, "v":pose_data[frame_idx][key]['v']}
    
    # Append the smoothed frame to the list
    smoothed_data[frame_idx] = smoothed_frame
    
  return smoothed_data