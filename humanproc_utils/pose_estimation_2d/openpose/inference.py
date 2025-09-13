import os

from .pose_tracker import run_posetracker


MIN_NUM_FRAMES = 25

def inference_video(video_file, staf_dir, display=True):

  # ========= Run tracking ========= #
  if not os.path.isabs(video_file):
    video_file = os.path.join(os.getcwd(), video_file)
  tracking_results = run_posetracker(video_file, staf_folder=staf_dir, display=display)
  
  # remove tracklets if num_frames is less than MIN_NUM_FRAMES
  for person_id in list(tracking_results.keys()):
    if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
      del tracking_results[person_id]

  return tracking_results