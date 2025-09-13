import warnings
warnings.filterwarnings('ignore')

import cv2 
import numpy as np
from .main import VitInference


def vitpose_inference(vid_file, model_name='weights/vitpose-b-multi-coco.pth', 
                        img_size = None, is_plot=True):
  
  cap = cv2.VideoCapture(vid_file)
  model = VitInference(model_name)

  frame_counter = 0
  results = {}

  while True:
    ret,frame = cap.read()
    if not ret:
      break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if img_size: frame = cv2.resize(frame, img_size)
    pts, tids, bboxes, drawn_frame, orig_frame = model.inference(frame, frame_counter)

    for id, p_ in enumerate(pts):
      p = np.hstack([p_[:,:2], np.ones((p_.shape[0],1))])
      confidence = p_[:,2]
      if str(id) in results:
        results[str(id)]["joints2d"].append(p)
        results[str(id)]["confidence"].append(confidence)
        results[str(id)]["frames"].append(frame_counter)
      else:
        results[str(id)] = {}
        results[str(id)]["joints2d"] = [p]
        results[str(id)]["confidence"] = [confidence]
        results[str(id)]["frames"] = [frame_counter]

    frame_counter += 1

    if is_plot:
      cv2.imshow("Video",drawn_frame)
      if cv2.waitKey(5) & 0xFF == 27:
        break


  cap.release()
  cv2.destroyAllWindows()

  for person_id in results:
    results[person_id] = {key: np.array(value) for key, value in results[person_id].items()}

  return results