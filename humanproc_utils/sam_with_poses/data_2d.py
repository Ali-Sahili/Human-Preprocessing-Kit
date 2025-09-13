

import numpy as np
from matplotlib.path import Path

from .utils import load_npz

#-----------------------------------------------------------------------------------------
def get_human_points_from_segments(polygon_points, step_size=2):
  # Step 1: Create a Path object from the polygon points
  polygon_path = Path(polygon_points)

  # Step 2: Determine the bounding box of the polygon
  min_x, min_y = np.min(polygon_points, axis=0)
  max_x, max_y = np.max(polygon_points, axis=0)

  # Step 3: Create a grid of points covering the bounding box
  x_values = np.arange(min_x, max_x + 1, step=step_size)
  y_values = np.arange(min_y, max_y + 1, step=step_size)
  grid_x, grid_y = np.meshgrid(x_values, y_values)
  grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

  # Step 4: Find points within the polygon using contains_points()
  inside_points = grid_points[polygon_path.contains_points(grid_points)]
  return inside_points

#-----------------------------------------------------------------------------------------
def prepare_2d_data_tmp(sam_results_file, vitpose_results_file, step_size=4, stop_frame=50):

  sam_results = load_npz(sam_results_file)
  vitpose_results = load_npz(vitpose_results_file)

  data_2d = {}
  for person_id, person_id_vit in zip(sam_results, vitpose_results):
    print(f" =============  {person_id}/{person_id_vit}  ================= ")
    segments = sam_results[person_id]["segments"]
    joints2d = vitpose_results[person_id_vit]["joints2d"]

    if stop_frame != -1:
      if len(segments) < stop_frame or len(joints2d) < stop_frame:
        continue
    for frame_idx in range(len(segments)):
      if frame_idx == stop_frame: break
      polygon_points = segments[frame_idx]
      joints2d_frame = joints2d[frame_idx]
      inside_points = get_human_points_from_segments(polygon_points, step_size=step_size)

      if person_id in data_2d:
        data_2d[person_id]["segments"].append(polygon_points)
        data_2d[person_id]["joints2d"].append(joints2d_frame[:, [1, 0]])
        data_2d[person_id]["human_pixels"].append(inside_points)
      else:
        data_2d[person_id] = {}
        data_2d[person_id]["segments"] = [polygon_points]
        data_2d[person_id]["joints2d"] = [joints2d_frame[:, [1, 0]]]
        data_2d[person_id]["human_pixels"] = [inside_points]

  return data_2d


#-----------------------------------------------------------------------------------------
def prepare_2d_data(results_file, step_size=4, stop_frame=50):

  results = load_npz(results_file)

  data_2d = {}
  for person_id in results:
    print(f" =============  PersonID: {person_id}  ================= ")
    segments = results[person_id]["segments"]
    joints2d = results[person_id]["joints2d"]

    if stop_frame != -1:
      if len(segments) < stop_frame or len(joints2d) < stop_frame:
        continue
    for frame_idx in range(len(segments)):
      if frame_idx == stop_frame: break
      polygon_points = segments[frame_idx]
      joints2d_frame = joints2d[frame_idx]
      inside_points = get_human_points_from_segments(polygon_points, step_size=step_size)

      if person_id in data_2d:
        data_2d[person_id]["segments"].append(polygon_points)
        data_2d[person_id]["joints2d"].append(joints2d_frame[:, [1, 0]])
        data_2d[person_id]["human_pixels"].append(inside_points)
      else:
        data_2d[person_id] = {}
        data_2d[person_id]["segments"] = [polygon_points]
        data_2d[person_id]["joints2d"] = [joints2d_frame[:, [1, 0]]]
        data_2d[person_id]["human_pixels"] = [inside_points]

  return data_2d