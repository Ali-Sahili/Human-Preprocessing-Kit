
import os
import cv2
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor

from .utils import setup_cfg, calculate_iou

#----------------------------------------------------------------------------------------
# Process video frame by frame
def segment_and_pose_video(video_path, yolo_model_path="weights/yolo11x-pose.pt", 
                            out_path = "", save_results=True, plot_during_processing=False):
    print(f"Processing the video at {video_path}")
    os.makedirs(out_path, exist_ok=True)

    # Initialize the predictor with the config
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    pose_estimator = YOLO(yolo_model_path)

    cap = cv2.VideoCapture(video_path)

    person_masks = {}
    person_bboxes = {}
    person_keypoints = {}
    person_keypoints_scores = {}
    person_frames = {}

    # To track the bounding box of each person across frames
    next_person_id = 0 
    person_bboxes_tmp = {}

    frame_count = 0
    visible_frame_count = {}
    while cap.isOpened():
        print(f" ============ Frame {frame_count} ============ ")
        ret, frame = cap.read()
        if not ret:
          break

        # Make predictions on the frame
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        # Filter out instances with a low score threshold
        mask = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()  # Bounding boxes (x1, y1, x2, y2)

        # Loop through instances to overlay the masks
        for i in range(len(mask)):
            if scores[i] > 0.9 and classes[i] == 0:
                bbox = boxes[i]
                mask_i = mask[i]

                # Match this detected person with a previous person based on IoU (bounding box overlap)
                matched_id = None
                for person_id, prev_bbox in person_bboxes_tmp.items():
                    iou = calculate_iou(bbox, prev_bbox)
                    if iou > 0.5:  # Threshold for matching (IoU > 0.5)
                        matched_id = person_id
                        break

                # If no match was found, assign a new ID
                if matched_id is None:
                    matched_id = f"person_{next_person_id}"
                    person_bboxes_tmp[matched_id] = bbox
                    next_person_id += 1

                # Store the mask for this person in the corresponding list
                if matched_id not in person_masks:
                    person_masks[matched_id] = []
                    person_bboxes[matched_id] = []
                    person_keypoints[matched_id] = []
                    person_keypoints_scores[matched_id] = []
                    person_frames[matched_id] = []

                    visible_frame_count[matched_id] = 0

                person_masks[matched_id].append(mask_i)
                person_bboxes[matched_id].append(bbox)
                person_frames[matched_id].append(visible_frame_count[matched_id])
                person_bboxes_tmp[matched_id] = bbox  # Update the person's bbox

                visible_frame_count[matched_id] += 1
                # Apply mask to isolate each person
                segmented_person = cv2.bitwise_and(frame, frame, mask=(mask_i * 255).astype(np.uint8))
                det_results = pose_estimator(segmented_person, verbose=False)
                for result in det_results:
                    kpts = result.keypoints.xy
                    kpts_scores = result.keypoints.conf
                    person_keypoints[matched_id].append(kpts.cpu().squeeze().numpy())
                    person_keypoints_scores[matched_id].append(kpts_scores.cpu().squeeze().numpy())
                    for kpt in kpts:
                      for x, y in kpt:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0), -1)

                # Create a 3-channel mask to overlay
                if plot_during_processing:
                  color_mask = np.zeros_like(frame)
                  color_mask[mask_i] = np.random.randint(0, 255, 3)
                  frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

        frame_count += 1
        if plot_during_processing:
          cv2.imshow('Segmented Frame', frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save results
    if save_results:
      print("Saving results...")
      frames, bboxes, masks, keypoints, keypoints_scores = [], [], [], [], []
      for key in person_masks:
        frames.append(person_frames[key])
        bboxes.append(person_bboxes[key])
        masks.append(person_masks[key])
        keypoints.append(np.array(person_keypoints[key]))
        keypoints_scores.append(np.array(person_keypoints_scores[key]))

      np.savez_compressed(os.path.join(out_path, '2d_masks_maskrcnn.npz'), np.array(masks, dtype=object), allow_pickle=True)
      np.savez_compressed(os.path.join(out_path, '2d_keypoints_yolopose.npz'), np.array(keypoints, dtype=object), allow_pickle=True)
      np.savez_compressed(os.path.join(out_path, '2d_keypoints_scores_yolopose.npz'), np.array(keypoints_scores, dtype=object), allow_pickle=True)
      np.savez_compressed(os.path.join(out_path, '2d_bboxes_maskrcnn.npz'), np.array(bboxes, dtype=object), allow_pickle=True)
      np.savez_compressed(os.path.join(out_path, 'visible_frames.npz'), np.array(frames, dtype=object), allow_pickle=True)
    
    print("Done.")