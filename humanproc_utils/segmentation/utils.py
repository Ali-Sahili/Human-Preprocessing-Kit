
import cv2
import numpy as np

from skimage.feature import corner_peaks
from skimage.morphology import skeletonize, binary_dilation

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

#-------------------------------------------------------------------------------
def skeletonize_mask(mask):
    """Applies skeletonization to the mask to obtain a stick figure."""
    binary_mask = (mask > 0).astype(np.uint8)    
    skeleton = skeletonize(binary_mask)
    return skeleton

#-------------------------------------------------------------------------------
def extract_keypoints_from_skeleton(skeleton):
  """Finds keypoints on the skeletonized mask."""
  # Dilate the skeleton to find corners and endpoints
  dilated_skeleton = binary_dilation(skeleton)

  # Detect corners as keypoints
  keypoints = corner_peaks(dilated_skeleton.astype(np.int16), min_distance=5)

  # Filter and organize keypoints based on proximity to common body landmarks
  # (e.g., top, middle, and bottom sections of the silhouette)
  
  return keypoints

#-------------------------------------------------------------------------------
def visualize_masks(frame, outputs, keypoints=None, meta_log='coco_2017_train'):
  # Visualize Masks and Keypoints
  v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(meta_log), scale=1.0)
  v = v.draw_instance_predictions(outputs.to("cpu"))
  seg_frame = v.get_image()[:, :, ::-1] # Convert to BGR for OpenCV display

  if isinstance(keypoints, np.ndarray):
    for x, y in keypoints[:, :2]:  # Only x, y coordinates
      cv2.circle(frame, (int(y), int(x)), radius=3, color=(0, 255, 0), thickness=-1)

  cv2.imshow('Result with poses', frame)
  cv2.imshow('Result with masks', seg_frame)
  if cv2.waitKey(5) & 0xFF == 27:
    cv2.destroyAllWindows()
    exit()