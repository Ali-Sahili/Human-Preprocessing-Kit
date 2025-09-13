import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from humanproc_utils.pose_estimation_2d.yolo.yolo_pose import yolo_inference


yolo_results = yolo_inference(
    video_path="inputs/sample_video.mp4",
    model_name="weights/yolo11x-pose.pt",
    is_plot_poses=True,
)
print(yolo_results)
