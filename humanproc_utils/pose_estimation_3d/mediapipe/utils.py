
import os
import json
import yaml
import numpy as np

#----------------------------------------------------------------------------------------------
class Config:
  def __init__(self, **entries):
    self.__dict__.update(entries)

#----------------------------------------------------------------------------------------------
def dict_to_config(d):
  if isinstance(d, dict):
    d = {k: dict_to_config(v) for k, v in d.items()}
    return Config(**d)
  else:
    return d
    
#----------------------------------------------------------------------------------------------
def load_yaml_as_config(file_path):
  with open(file_path, 'r') as file:
      config_dict = yaml.safe_load(file)
  return dict_to_config(config_dict)

#----------------------------------------------------------------------------------------------
def load_pose_data(json_path):
  print(" ================== Loading Keypoints ================== ")
  with open(json_path, 'r') as file:
      pose_data = json.load(file)
  return pose_data

#----------------------------------------------------------------------------------------------
def load_numpy_pose_data(npy_path):
  return np.load(npy_path)

#----------------------------------------------------------------------------------------------
def save_dict_as_json(data, name="tmp"):
  with open(f'{name}', 'w') as f:
    json.dump(data, f, indent=4)

#----------------------------------------------------------------------------------------------
# Save data as JSON file
def save_keypoints(pose_data, output_folder, name="poses"):
  print(" ================= Keypoints Saving ================= ")
  os.makedirs(output_folder, exist_ok=True)
  filename = check_file_exists(os.path.join(output_folder, name + f".json"), suffix="json")
  save_dict_as_json(pose_data, name=filename)

#----------------------------------------------------------------------------------------------
def save_numpy_keypoints(pose_data, output_folder, name="poses"):
  print(" ================= Saving Numpy Keypoints =========== ")
  os.makedirs(output_folder, exist_ok=True)
  filename = check_file_exists(os.path.join(output_folder, name + f".npy"), suffix="npy")
  np.save(filename,  pose_data)

#----------------------------------------------------------------------------------------------
def check_file_exists(filename, suffix="json"):
  if os.path.isfile(filename):
    expand = 1
    while True:
      expand += 1
      new_filename = filename.split(".")[0] + str(expand) + f".{suffix}"
      if os.path.isfile(new_filename):
        continue
      else:
        filename = new_filename
        break
  return filename

#----------------------------------------------------------------------------------------------
def get_min_max_list_of_dicts(data):
  if not isinstance(data, dict) or not bool(data):
    return []

  xs, ys, zs = [], [], []
  for pose in list(data.values()):
    xs = xs + [l['x'] for l in list(pose.values())]
    ys = ys + [l['y'] for l in list(pose.values())]
    zs = zs + [l['z'] for l in list(pose.values())]
  return [min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)]

#----------------------------------------------------------------------------------------------
def get_value_by_index(l, idx, mode): # mode="min" / mode="max"
  if len(l) > 0:
    return l[idx]
  return 999 if mode == "min" else -999

#----------------------------------------------------------------------------------------------
def get_min_max_per_keypoints(keypoints):
  mm_ps = get_min_max_list_of_dicts(keypoints["poses"])
  mm_lh = get_min_max_list_of_dicts(keypoints["left_hand"])
  mm_rh = get_min_max_list_of_dicts(keypoints["right_hand"])
  mm_face = get_min_max_list_of_dicts(keypoints["face"])

  min_x = min(get_value_by_index(mm_ps, 0, mode="min"), get_value_by_index(mm_lh, 0, mode="min"),
              get_value_by_index(mm_rh, 0, mode="min"), get_value_by_index(mm_face, 0, mode="min") )
  max_x = max(get_value_by_index(mm_ps, 1, mode="max"), get_value_by_index(mm_lh, 1, mode="max"),
              get_value_by_index(mm_rh, 1, mode="max"), get_value_by_index(mm_face, 1, mode="max") )
  min_y = min(get_value_by_index(mm_ps, 2, mode="min"), get_value_by_index(mm_lh, 2, mode="min"),
              get_value_by_index(mm_rh, 2, mode="min"), get_value_by_index(mm_face, 2, mode="min") )
  max_y = max(get_value_by_index(mm_ps, 3, mode="max"), get_value_by_index(mm_lh, 3, mode="max"),
              get_value_by_index(mm_rh, 3, mode="max"), get_value_by_index(mm_face, 3, mode="max") )
  min_z = min(get_value_by_index(mm_ps, 4, mode="min"), get_value_by_index(mm_lh, 4, mode="min"),
              get_value_by_index(mm_rh, 4, mode="min"), get_value_by_index(mm_face, 4, mode="min") )
  max_z = max(get_value_by_index(mm_ps, 5, mode="max"), get_value_by_index(mm_lh, 5, mode="max"),
              get_value_by_index(mm_rh, 5, mode="max"), get_value_by_index(mm_face, 5, mode="max") )
  
  return min_x, max_x, min_y, max_y, min_z, max_z

#----------------------------------------------------------------------------------------------
def dict_to_lists_of_coords(poses):
  if not isinstance(poses, dict) or not bool(poses):
    return None, []
  
  list_coords = []
  list_frame_inds = []
  for frame_idx, pose_frame in poses.items():
    poses_per_frame = []
    for l_name, l_coords in pose_frame.items():
      poses_per_frame.append([l_coords['x'], l_coords['y'], l_coords['z']])
    list_coords.append(poses_per_frame)
    list_frame_inds.append(int(frame_idx))
  return np.asarray(list_coords, dtype=np.float32), np.array(list_frame_inds)

#----------------------------------------------------------------------------------------------
def is_empty_numpy(np_array):
  return np_array.shape[0] == 0