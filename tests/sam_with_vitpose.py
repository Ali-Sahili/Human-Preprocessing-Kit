import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import numpy as np
from humanproc_utils.segmentation.sam import sam_inference
from humanproc_utils.pose_estimation_2d.vitpose.inference import vitpose_inference

from humanproc_utils.sam_with_poses.utils import save_dict_as_npz
from humanproc_utils.sam_with_poses.data_2d import prepare_2d_data_tmp
from humanproc_utils.sam_with_poses.visualize import plot_2d_data


# -----------------------------------------------------------------------------------------

video_name = "two_people"  # "sample_video"
sam_results_file = f"outputs/sam/{video_name}/results.npz"
vitpose_results_file = f"outputs/vitpose/{video_name}/results.npz"
video_path = f"inputs/{video_name}.mp4"
is_plot = False

if not os.path.exists(vitpose_results_file):
    vitpose_results = vitpose_inference(
        video_path, model_name="weights/vitpose-b-multi-coco.pth", is_plot=is_plot
    )
    save_dict_as_npz(vitpose_results, f"outputs/vitpose/{video_name}")

if not os.path.exists(sam_results_file):
    sam_results = sam_inference(
        video_path,
        det_model="weights/yolo11x.pt",
        sam_model="weights/sam_b.pt",
        device="cuda",
        is_plot=is_plot,
        save_img=False,
        nb_max_frames=-1,
    )

    save_dict_as_npz(sam_results, f"outputs/sam/{video_name}")


data_2d = prepare_2d_data_tmp(sam_results_file, vitpose_results_file, step_size=4)
print("2D Data is ready.")

for person_id in data_2d:
    segments = data_2d[person_id]["segments"]
    joints2d = data_2d[person_id]["joints2d"]
    inside_points = data_2d[person_id]["human_pixels"]
    plot_2d_data(segments, joints2d, inside_points, marging=25)
    print(" ================================================== ")
