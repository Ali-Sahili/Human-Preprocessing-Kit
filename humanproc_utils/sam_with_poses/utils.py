
import os
import numpy as np


#-----------------------------------------------------------------------------------------
def save_dict_as_npz(data, folder, filename="results"):
  os.makedirs(folder, exist_ok=True)
  np.savez(os.path.join(folder, filename+".npz"),data)

#-----------------------------------------------------------------------------------------
def load_npz(file_name):
  return np.load(file_name, allow_pickle=True)["arr_0"].item()

#-----------------------------------------------------------------------------------------
def get_min_max(list_nps):
  global_min = np.array([float('inf'), float('inf')])
  global_max = np.array([float('-inf'), float('-inf')])

  for arr in list_nps:
    min_per_dim = np.min(arr, axis=0)
    max_per_dim = np.max(arr, axis=0)
    
    global_min = np.minimum(global_min, min_per_dim)
    global_max = np.maximum(global_max, max_per_dim)
  return global_min, global_max