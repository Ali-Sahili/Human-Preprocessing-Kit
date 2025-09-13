import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import torch
import numpy as np
from humanproc_utils.segmentation.deeplabv3 import deeplabv3_predictor
from humanproc_utils.segmentation.utils import (
    skeletonize_mask,
    extract_keypoints_from_skeleton,
)


model, preprocess = deeplabv3_predictor()


cap = cv2.VideoCapture("inputs/sample_video.mp4")
frame_count = 0
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Segment humans
    input_tensor = preprocess(frame).unsqueeze(0)  # frame is an image
    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    person_mask = (output.argmax(0) == 15).byte().cpu().numpy()

    skeleton = skeletonize_mask(person_mask)
    keypoints = extract_keypoints_from_skeleton(skeleton)

    for x, y in keypoints[:, :2]:  # Only x, y coordinates
        cv2.circle(frame, (int(y), int(x)), radius=3, color=(0, 0, 255), thickness=-1)

    color_mask = np.zeros(
        (person_mask.shape[0], person_mask.shape[1], 3), dtype=np.uint8
    )
    color_mask[person_mask == 1] = [0, 255, 0]  # Green color for "person" mask

    frame_np = np.array(frame)

    alpha = 0.5  # Transparency factor
    overlay = frame_np.copy()
    overlay[color_mask[:, :, 1] == 255] = [
        0,
        255,
        0,
    ]  # Only set green where mask is True
    blended = cv2.addWeighted(frame_np, 1 - alpha, overlay, alpha, 0)

    # Save or display result
    cv2.imshow("Person Segmentation", blended)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    frame_count += 1
    print(f"Frame {frame_count}")

cap.release()
cv2.destroyAllWindows()
