
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches


#-----------------------------------------------------------------------------------------
def plot_results(values, add_boxes=True, add_poses=True):
    masks = values["masks"]
    joints2d = values["kpts"]
    bboxes = values["bboxes"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = colors.ListedColormap(['white', 'red'])

    nb_frames = len(masks)
    for frame_idx, (mask, kpt, bbox) in enumerate(zip(masks, joints2d, bboxes)):
        ax.cla()
        ax.imshow(mask, cmap=cmap, alpha=0.5)

        if add_poses: ax.scatter(kpt[:, 0], kpt[:, 1], c="r", s=10)

        if add_boxes:
            x = int(bbox[0]); y = int(bbox[1])
            w = int(bbox[2]-bbox[0]); h = int(bbox[3]-bbox[1])
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f"Masks and Poses - Frame {frame_idx+1}/{nb_frames}")
        plt.axis('off')
        plt.pause(0.05)
    plt.show()