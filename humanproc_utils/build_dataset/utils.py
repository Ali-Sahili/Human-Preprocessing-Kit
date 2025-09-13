
import os
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo


#----------------------------------------------------------------------------------------
# Set up Detectron2 configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # to filter predictions
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

#----------------------------------------------------------------------------------------
# Helper function to calculate distance between bounding boxes
def calculate_iou(bb1, bb2):
    # Calculate the intersection over union (IoU) of two bounding boxes
    x1, y1, x2, y2 = bb1
    x1_2, y1_2, x2_2, y2_2 = bb2
    
    # Calculate the coordinates of the intersection rectangle
    ix1 = max(x1, x1_2)
    iy1 = max(y1, y1_2)
    ix2 = min(x2, x2_2)
    iy2 = min(y2, y2_2)
    
    # Calculate area of intersection
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    # Calculate area of both bounding boxes
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate the area of union
    union_area = bb1_area + bb2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

#----------------------------------------------------------------------------------------
def save_dict_as_npz_compressed(data, folder, filename="results"):
  os.makedirs(folder, exist_ok=True)
  if not filename.endswith('.npz'):
    filename += ".npz"
  np.savez_compressed(os.path.join(folder, filename),data)

#----------------------------------------------------------------------------------------
def load_npz_compressed_dict(file_name):
  return np.load(file_name, allow_pickle=True)["arr_0"].item()

#----------------------------------------------------------------------------------------
def load_npz_compressed(file_name):
  return np.load(file_name, allow_pickle=True)["arr_0"]