import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from humanproc_utils.pose_estimation_2d.openpose.inference import inference_video


tracking_results = inference_video(
    video_file="inputs/sample_video.mp4", staf_dir="/home/maxali/STAF/", display=True
)
