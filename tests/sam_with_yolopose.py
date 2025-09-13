import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os

from humanproc_utils.sam_with_poses.utils import save_dict_as_npz
from humanproc_utils.sam_with_poses.seg_and_pose import (
    segmentation_with_pose_estimation,
)
from humanproc_utils.sam_with_poses.visualize import plot_2d_data
from humanproc_utils.sam_with_poses.data_2d import prepare_2d_data


video_name = "sample_video"  # "sample_video"
res_path = f"outputs/seg_and_pose/{video_name}/"
results_filename = "results"
video_path = f"inputs/{video_name}.mp4"
is_plot = True

if not os.path.exists(os.path.join(res_path, f"{results_filename}.npz")):
    results = segmentation_with_pose_estimation(
        video_path=video_path,
        seg_model="weights/sam_b.pt",
        yolo_model="weights/yolo11x-pose.pt",
        is_plot=is_plot,
    )

    save_dict_as_npz(results, res_path, filename=results_filename)


results_file = os.path.join(res_path, f"{results_filename}.npz")
data2d = prepare_2d_data(results_file, step_size=4, stop_frame=-1)

for person_id in data2d:
    segments = data2d[person_id]["segments"]
    joints2d = data2d[person_id]["joints2d"]
    inside_points = data2d[person_id]["human_pixels"]
    plot_2d_data(segments, joints2d, inside_points, marging=25)
    print(" ============================================= ")
    assert 0
