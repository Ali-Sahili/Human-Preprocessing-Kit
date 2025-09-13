
import torch
import logging

from .tracker import byte_tracker
from .model_builder import build_model
from .pose_utils.timerr import Timer
from .pose_utils.visualizer import plot_tracking
from .pose_utils.general_utils import make_parser
from .pose_utils.pose_utils import pose_points_yolo5
from .pose_utils.logger_helper import CustomFormatter
from .pose_utils.pose_viz import draw_points_and_skeleton, joints_dict



logger = logging.getLogger("Tracker !")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.propagate=False


class VitInference:
  # tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
  def __init__(self,pose_path):
    super(VitInference,self).__init__()
    self.tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
    self.pose_path = pose_path
    print(self.pose_path)

    self.model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5n', pretrained=True)
    self.pose = build_model('ViTPose_base_coco_256x192',self.pose_path)
    self.pose.cuda().eval()
    self.timer = Timer()

  def inference(self,img,frame_id=0):
    frame_orig = img.copy()
    self.timer.tic()
    pts,online_tlwhs,online_ids,online_scores = pose_points_yolo5(self.model, img, self.pose, self.tracker)

    self.timer.toc()
    if len(online_ids)>0:
      # timer_track.tic()
      # self.timer.tic()
      online_im = frame_orig.copy()
      online_im = plot_tracking(
          frame_orig, online_tlwhs, online_ids, frame_id=frame_id, fps=1/self.timer.average_time
      )
      # self.timer.toc()
      if pts is not None:
        for i, (pt, pid) in enumerate(zip(pts, online_ids)):
          online_im=draw_points_and_skeleton(online_im, pt, joints_dict()['coco']['skeleton'], 
                                                person_index=pid,
                                                points_color_palette='gist_rainbow', 
                                                skeleton_color_palette='jet',
                                                points_palette_samples=10,
                                                confidence_threshold=0.3
                                          )
    else:
      online_im = frame_orig
        
    return pts,online_ids,online_tlwhs,online_im,frame_orig



