
import cv2
import numpy as np


def expand_mask(mask, nb_iters=3):
  kernel = np.ones((3, 3), np.uint8)
  expanded_mask = cv2.dilate(mask, kernel, iterations=nb_iters)
  return expanded_mask

def visualize_difference(original_mask, expanded_mask):
  """
  Combines the original and expanded masks into a single image to visualize the difference.
  """
  height, width = original_mask.shape
  visual = np.zeros((height, width, 3), dtype=np.uint8)

  visual[original_mask == 255] = [255, 255, 255]  # Red for original mask
  # visual[expanded_mask == 255] = [255, 0, 0]  # Blue for expanded mask
  visual[(original_mask == 0) & (expanded_mask == 255)] = [255, 0, 255]  # Purple for overlap

  return visual

if __name__ == "__main__":
  from utils import load_npz_compressed
  masks = load_npz_compressed(file_name="../outputs/dataset/sample_video/2d_masks_maskrcnn.npz")
  for mask_per_person in masks:
    for mask_per_frame in mask_per_person:

      _, mask = cv2.threshold((mask_per_frame * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
      expanded_mask = expand_mask(mask)

      result = visualize_difference((mask_per_frame * 255).astype(np.uint8), expanded_mask)
      cv2.imshow("Expanded Mask", result)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    break
  cv2.destroyAllWindows()