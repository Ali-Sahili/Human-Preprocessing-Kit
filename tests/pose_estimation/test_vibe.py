import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from humanproc_utils.pose_estimation_3d.vibe.lib.utils.utils import read_pkl_file
from humanproc_utils.pose_estimation_3d.vibe.lib.inference import vibe_infernce_video
from humanproc_utils.pose_estimation_3d.vibe.lib.visualize import plot_3d_pose_sequence
from humanproc_utils.pose_estimation_3d.vibe.lib.prepare_output import prepare_keypoints
from humanproc_utils.pose_estimation_3d.vibe.lib.utils.utils import (
    load_yaml_as_config,
    print_dict,
)

# VIBE Output
# -------------
# 'pred_cam',
# 'orig_cam',
# 'verts',               (frames, 6890, 3)
# 'pose',                (frames, 72) # global body rotation and the relative rotation
#                                       of 23 joints in axis-angle format.
# 'betas',               (frames, 10) # the first 10 coefficients of a PCA shape space
# 'joints3d',            (frames, 49, 3)
# 'joints2d',
# 'joints2d_img_coord',
# 'bboxes',
# 'frame_ids'            (frames, )


if __name__ == "__main__":

    cfg = load_yaml_as_config("configs/pose_estimation/vibe.yaml")
    cfg_pose = cfg.pose_estimation
    input_path = cfg_pose.load_pkl_path

    if not os.path.isfile(input_path):
        vibe_infernce_video(cfg_pose)

    else:
        vibe_output = read_pkl_file(input_path)
        print("NUmber of detected persons: ", len(vibe_output))

        if len(vibe_output) > 0:
            for id in vibe_output.keys():
                print(f"Person ID: {id}")
                if cfg_pose.plot_skeleton:
                    plot_3d_pose_sequence(
                        vibe_output[id]["joints3d"],
                        vibe_output[id]["verts"],
                        add_mesh=True,
                        add_fingers=True,
                        frame_step=1,
                        nb_frames_to_visualize=0,
                    )

                # keypoints = prepare_keypoints(vibe_output[id]['joints3d'], vibe_output[id]['verts'])
                # print_dict(keypoints[0])
                # print(vibe_output[id]['frame_ids'])
