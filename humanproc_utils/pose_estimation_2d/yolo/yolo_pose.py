
import cv2
import numpy as np
from ultralytics import YOLO



def yolo_inference(video_path = "inputs/samurai.mp4", model_name="yolo11x-pose.pt",
                          is_plot_poses = True):
  
  model = YOLO(model_name)  # n,s,m,l,x
  cap = cv2.VideoCapture(video_path)
  
  frame_count = 0
  bbox, bbox_normalized, classes, bbox_conf, org_img_shape = [], [], [], [], []
  keypoints, keypoints_normalized, keypoints_conf, frame_ids = [], [], [], []
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break

    results = model(frame, verbose = False)
    
    # ----------------------------------------------------------------------------
    keypoints_, keypoints_normalized_, keypoints_conf_ = [], [], []
    for kk in results[0].keypoints:
      keypoints_.append(kk.xy.cpu().squeeze().numpy())
      keypoints_normalized_.append(kk.xyn.cpu().squeeze().numpy())
      keypoints_conf_.append(kk.conf.cpu().squeeze().numpy())

    keypoints.append(np.asarray(keypoints_, dtype=np.float32))
    keypoints_normalized.append(np.asarray(keypoints_normalized_, dtype=np.float32))
    keypoints_conf.append(np.asarray(keypoints_conf_, dtype=np.float32))
    
    bbox.append(results[0].boxes.xywh.cpu().numpy())
    bbox_normalized.append(results[0].boxes.xywhn.cpu().numpy())
    classes.append(results[0].boxes.cls.cpu().numpy())
    bbox_conf.append(results[0].boxes.conf.cpu().numpy())
    org_img_shape = results[0].orig_shape
    frame_ids.append(frame_count)

    frame_count += 1

    if is_plot_poses:
      annotated_frame = results[0].plot(boxes = False)  
      cv2.imshow("results", annotated_frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  cap.release()
  cv2.destroyAllWindows()

  return {
      "joints2d": keypoints,
      "joints2d_normalized": keypoints_normalized,
      "joints2d_conf": keypoints_conf,
      "class": classes,
      "bbox": bbox,
      "bbox_normalized": bbox_normalized,
      "bbox_conf": bbox_conf,
      "frames": frame_ids,
      "org_img_shape": org_img_shape
  }