import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2

from humanproc_utils.segmentation.maskrcnn import maskrcnn_predictor

from humanproc_utils.segmentation.utils import visualize_masks
from humanproc_utils.segmentation.utils import (
    skeletonize_mask,
    extract_keypoints_from_skeleton,
)


predictor, meta_log = maskrcnn_predictor()

cap = cv2.VideoCapture("inputs/sample_video.mp4")
frame_count = 0
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # For each frame in the video:
    outputs = predictor(frame)["instances"]
    human_masks = outputs.pred_masks[outputs.pred_classes == 0]  # 0 for person class

    skeleton = skeletonize_mask(human_masks[0].cpu().numpy().astype(int))
    keypoints = extract_keypoints_from_skeleton(skeleton)

    visualize_masks(frame, outputs, keypoints=keypoints, meta_log=meta_log)

    frame_count += 1
    print(f"Frame {frame_count}")

cap.release()
