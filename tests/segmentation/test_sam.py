import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from humanproc_utils.segmentation.sam import sam_inference


results = sam_inference(
    video_path="inputs/sample_video.mp4",
    det_model="weights/yolo11x.pt",  # "yolov8x.pt"
    sam_model="weights/sam_b.pt",
    device="",
    is_plot=True,
    save_img=False,
)
