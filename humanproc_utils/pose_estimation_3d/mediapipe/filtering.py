
from collections import OrderedDict
from .constants import head_body_indices, lower_body_indices, upper_body_indices


# Initialize dictionary to store "frozen" positions

frozen_lower_body_coords = {name: None for name in lower_body_indices}
first_time_dict = {name: True for name in lower_body_indices}

lower_body_indices = {  "LEFT_HIP": (-0.05, 0.2, 0.0), # (0.0, 0.2, 0.0),
                        "RIGHT_HIP": (0.05, 0.2, 0.0), # (0.1, 0.2, 0.0),
                        "LEFT_KNEE": (-0.05, 0.4, 0.0), # (0.0, 0.4, 0.0),
                        "RIGHT_KNEE": (0.05, 0.4, 0.0), # (0.1, 0.4, 0.0),
                        "LEFT_ANKLE": (-0.05, 0.6, 0.0), # (0.0, 0.6, 0.0),
                        "RIGHT_ANKLE": (0.05, 0.6, 0.0), # (0.1, 0.6, 0.0),
                        "LEFT_FOOT_INDEX": (-0.05, 0.8, 0.0), # (0.0, 0.7, 0.0),
                        "RIGHT_FOOT_INDEX": (0.05, 0.8, 0.0), # (0.1, 0.7, 0.0),
                    }

left_offset_x = -0.05  # Move left-side points slightly left
right_offset_x = 0.05  # Move right-side points slightly right
depth_offset_z = 0.02  # Optional: Separate depth to avoid z-axis overlap

#------------------------------------------------------------------------------------------
def filter_unseen_parts(pose_data, offset=0.2):
  print(" ================= Filtering ======================== ")
  new_pose_data = {}
  for frame_idx, pose_frame in pose_data.items():
    new_pose_frame = OrderedDict()
    
    mid_hip_x = (pose_frame["LEFT_HIP"]['x'] + pose_frame["RIGHT_HIP"]['x']) / 2
    mid_hip_y = (pose_frame["LEFT_HIP"]['y'] + pose_frame["RIGHT_HIP"]['y']) / 2
    mid_hip_z = (pose_frame["LEFT_HIP"]['z'] + pose_frame["RIGHT_HIP"]['z']) / 2
    
    for l_name, landmark in pose_frame.items():
      visibility = landmark['v']

      # Only adjust lower body landmarks with low visibility
      if l_name in lower_body_indices.keys() and visibility < 0.5:
        if frozen_lower_body_coords[l_name] is not None:
          # If we have a frozen position from previous frames, use it
          landmark['x'], landmark['y'], landmark['z'] = frozen_lower_body_coords[l_name]
        else:
          # Set a relative offset based on mid-hip coordinates
          offset_x, offset_y, offset_z = lower_body_indices[l_name]
          landmark['x'] = mid_hip_x + offset_x
          landmark['y'] = mid_hip_y + offset_y
          landmark['z'] = mid_hip_z + offset_z

      #---------------------------------------------------------------------------------
      # if l_name == "LEFT_FOOT_INDEX":
      #   landmark['z'] += 0.05  # Move left foot forward in z-axis
      # elif l_name == "RIGHT_FOOT_INDEX":
      #   landmark['z'] -= 0.05  # Move right foot backward in z-axis

      #---------------------------------------------------------------------------------
      # to solve overlapping 
      if l_name in ["LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"]:
        # Apply left-side offset to x (and optionally z)
        landmark['x'] += left_offset_x
        landmark['z'] += depth_offset_z  # Optional: adjust z for better separation
    
      elif l_name in ["RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"]:
        # Apply right-side offset to x (and optionally z)
        landmark['x'] += right_offset_x
        landmark['z'] -= depth_offset_z 

      # Update frozen coordinates if visibility is sufficient
      if l_name in lower_body_indices.keys() and visibility >= 0.5:
        frozen_lower_body_coords[l_name] = (landmark['x'], landmark['y'], landmark['z'])

    #---------------------------------------------------------------------------------
    # for l_name, landmark in pose_frame.items():
    #   visibility = landmark["v"]
    
    #   # Check if current landmark is a lower-body keypoint
    #   if l_name in lower_body_indices:
    #     if visibility >= 0.5:
    #       # If visible, update the frozen position
    #       frozen_lower_body_coords[l_name] = {'x': landmark['x'], 'y': landmark['y'], 'z': landmark['z'], 'v': landmark["v"]}
    #     elif bool(frozen_lower_body_coords[l_name]):
    #       # If not visible, use the frozen position
    #       landmark['x'] = frozen_lower_body_coords[l_name]['x']
    #       landmark['y'] = frozen_lower_body_coords[l_name]['y']
    #       landmark['z'] = frozen_lower_body_coords[l_name]['z']
    #     else:
    #       # If no good visibility frames yet, set a reasonable default position
    #       # Align to average upper-body landmarks for a reasonable estimation
    #       if first_time_dict[l_name]:
    #         first_time_dict[l_name] = False
    #         frozen_lower_body_coords[l_name] = {'x': landmark['x'], 'y': landmark['y'], 'z': landmark['z'], 'v': landmark["v"]}
    #       else:
    #         print("12345667890--===8723984723749823749827398472983749827398472")
    #         align_with_l, align_with_r = "LEFT_HIP", "RIGHT_HIP"
    #         avg_shoulder_x = (pose_frame[align_with_l]['x']+pose_frame[align_with_r]['x'])/2
    #         avg_shoulder_y = (pose_frame[align_with_l]['y']+pose_frame[align_with_r]['y'])/2
    #         avg_shoulder_z = (pose_frame[align_with_l]['z']+pose_frame[align_with_r]['z'])/2
    #         landmark["x"], landmark["y"] = avg_shoulder_x, avg_shoulder_y + offset  # Slight offset below shoulders
    #         landmark["z"] = avg_shoulder_z + 1.

      new_pose_frame[l_name] = landmark
    new_pose_data[frame_idx] = new_pose_frame
 
  return new_pose_data