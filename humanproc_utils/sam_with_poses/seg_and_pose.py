
import cv2
import numpy as np
from ultralytics import SAM, YOLO


def segmentation_with_pose_estimation(  video_path = 'inputs/samurai.mp4',
                                        seg_model="weights/sam_b.pt", 
                                        yolo_model="weights/yolo11x-pose.pt", is_plot = True):
  pose_estimator = YOLO(yolo_model)
  sam_model = SAM(seg_model)

  cap = cv2.VideoCapture(video_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  outputs = {}
  frame_count = 0
  while cap.isOpened():
    print(f" =========== SAM+YOLO-POSE | Frame {frame_count} =========== ")
    ret, frame = cap.read()
    if not ret:
        break

    det_results = pose_estimator(frame, verbose=False)

    for result in det_results:
      class_ids = result.boxes.cls.int().tolist()
      kpts = result.keypoints.xy
      kpts_n = result.keypoints.xyn
      if len(class_ids):
        boxes = result.boxes.xyxy
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device="")
        segments = sam_results[0].masks.xy
        segments_n = sam_results[0].masks.xyn  # noqa
        
        for idx, (segment, kpt) in enumerate(zip(segments, kpts)):
          if len(segment) == 0:
              continue
          if class_ids[idx] == 0:
            key = str(idx)
            if key in outputs:
              outputs[key]["frames"].append(frame_count)
              outputs[key]["bbox"].append(boxes[idx].cpu().numpy())
              outputs[key]["segments"].append(segment)
              outputs[key]["joints2d"].append(kpt.cpu().numpy())
              outputs[key]["segments_n"].append(segments_n[idx])
              outputs[key]["joints2d_n"].append(kpts_n[idx].cpu().numpy())
            else:
              outputs[key] = {}
              outputs[key]["frames"] = [frame_count]
              outputs[key]["bbox"] = [boxes[idx].cpu().numpy()]
              outputs[key]["segments"] = [segment]
              outputs[key]["joints2d"] = [kpt.cpu().numpy()[:, [1,0]]]
              outputs[key]["segments_n"] = [segments_n[idx]]
              outputs[key]["joints2d_n"] = [kpts_n[idx].cpu().numpy()[:, [1,0]]]
            if is_plot:
              cv2.polylines(frame, np.int32([segment]), True, (0, 255, 0), 2)
              cv2.fillPoly(frame, np.int32([segment]), (255, 0, 0))
              for x, y in kpt:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)


    frame_count += 1
    if is_plot:
      cv2.imshow("results", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  cap.release()
  cv2.destroyAllWindows()
  return outputs