
import cv2
import numpy as np
from ultralytics import SAM, YOLO



def sam_inference(video_path, det_model="yolov8x.pt", sam_model="sam_b.pt",  device="", 
                        is_plot=True, save_img=False, nb_max_frames=-1):

  det_model = YOLO(det_model)
  sam_model = SAM(sam_model)

  cap = cv2.VideoCapture(video_path)

  frame_count = 0
  sam_output = {}
  while cap.isOpened():
    print(f"SAM     | Frame {frame_count}")
    if nb_max_frames == frame_count:
      break
    success, frame = cap.read()
    if not success:
        break

    det_results = det_model(frame, stream=True, device=device)

    for result in det_results:
      class_ids = result.boxes.cls.int().tolist()
      if len(class_ids):
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=save_img, device=device)
        segments = sam_results[0].masks.xy
        segments_n = sam_results[0].masks.xyn  # noqa

        for idx, segment in enumerate(segments):
          if len(segment) == 0:
              continue
          if class_ids[idx] == 0:
            key = str(idx)
            if key in sam_output:
              sam_output[key]["frames"].append(frame_count)
              sam_output[key]["bbox"].append(boxes[idx].cpu().numpy())
              sam_output[key]["segments"].append(segment)
              sam_output[key]["segments_n"].append(segments_n[idx])
            else:
              sam_output[key] = {}
              sam_output[key]["frames"] = [frame_count]
              sam_output[key]["bbox"] = [boxes[idx].cpu().numpy()]
              sam_output[key]["segments"] = [segment]
              sam_output[key]["segments_n"] = [segments_n[idx]]
            if is_plot:
              cv2.polylines(frame, np.int32([segment]), True, (0, 255, 0), 2)
              cv2.fillPoly(frame, np.int32([segment]), (255, 0, 0))

    frame_count += 1
    if is_plot:
      cv2.imshow("results", frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  print(" ============================== ")
  return sam_output