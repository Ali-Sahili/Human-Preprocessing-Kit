import warnings

warnings.filterwarnings("ignore")

import os
import glob

from humanproc_utils.build_dataset.builder import segment_and_pose_video
from humanproc_utils.build_dataset.video_masking import save_masked_videos_per_person

from humanproc_utils.build_dataset.PoseFormerV2.visualize import img2video
from humanproc_utils.build_dataset.PoseFormerV2.poses_processing import (
    get_pose2D,
    get_pose3D,
)

# --------------------------------------------------------------------------------------
video_name = "sample_video.mp4"  # "two_people.mp4" # "running_man_2sec.mp4" # "dancing_people.mp4" #
yolo_model_path = "weights/yolo11x-pose.pt"


video_suffix = video_name.split(".")[-1]
video_name_without_suffix = video_name.split(".")[0]
video_path = f"inputs/{video_name}"
out_path = f"outputs/dataset/{video_name_without_suffix}/"
args_gpu = "0"

segment_and_pose_video(
    video_path,
    yolo_model_path,
    out_path=out_path,
    save_results=True,
    plot_during_processing=False,
)
save_masked_videos_per_person(video_path, out_path, plot_during_processing=False)

print(" ========================================== ")
print("Estimating 3d poses: ")
print()

masked_videos_paths = sorted(
    glob.glob(os.path.join(out_path + "masked_videos/", f"*.{video_suffix}"))
)
os.environ["CUDA_VISIBLE_DEVICES"] = args_gpu
for vid_path in masked_videos_paths:
    out_path_ = os.path.join(
        out_path,
        os.path.join("masked_video_poses", vid_path.split("/")[-1].split(".")[0]),
    )
    out_path_ += "/"
    keypoints, scores = get_pose2D(vid_path, out_path_)
    get_pose3D(vid_path, out_path_, keypoints, scores)
    img2video(vid_path, out_path_)
