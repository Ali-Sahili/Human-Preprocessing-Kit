import os
import cv2
import numpy as np

from .utils import load_npz_compressed

NUM_MIN_FRAMES = 20

#--------------------------------------------------------------------------------------
def get_masked_video(video_path, video_out_path, masks, frames, plot_during_processing):
    cap = cv2.VideoCapture(video_path)

    # Output video writers
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cap = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))


    frame_count = 0
    while cap.isOpened():
        # print(f" ============ Frame {frame_count} ============ ")
        ret, frame = cap.read()
        if not ret:
          break

        if frame_count in frames:
          segmented_person = cv2.bitwise_and(frame, frame, mask=(masks[frame_count] * 255).astype(np.uint8))
          out_cap.write(segmented_person)

          if plot_during_processing:
            cv2.imshow('Segmented Frame', segmented_person)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        elif frame_count not in frames:
          print("skip frame...")
        
        frame_count += 1

    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------
def save_masked_videos_per_person(video_path, out_path, plot_during_processing=False):
  video_suffix = video_path.split(".")[-1]
  video_name_without_suffix = (video_path.split("/")[-1]).split(".")[0]
  masks = load_npz_compressed(file_name=os.path.join(out_path, "2d_masks_maskrcnn.npz"))
  frames = load_npz_compressed(file_name=os.path.join(out_path, "visible_frames.npz"))

  out_path += "masked_videos"
  os.makedirs(out_path, exist_ok=True)
  for person_id, mask_per_person in enumerate(masks):
    if len(mask_per_person) < NUM_MIN_FRAMES:
      print(f"Skipping video for person {person_id}. Unsufficient Number of frames !")
      continue
    print(f"Saving video of Person {person_id}")
    frames_per_person = frames[person_id]

    get_masked_video(video_path, 
                        os.path.join(out_path,
                            f"masked_{video_name_without_suffix}_{person_id}.{video_suffix}"),
                        mask_per_person, 
                        frames_per_person, 
                        plot_during_processing
                )
  print("Done.")