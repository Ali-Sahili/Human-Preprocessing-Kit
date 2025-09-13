import warnings

warnings.filterwarnings("ignore")

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from humanproc_utils.pose_estimation_3d.mediapipe import save_numpy_keypoints
from humanproc_utils.pose_estimation_3d.mediapipe import (
    visualize_3d_skeleton_numpy,
    plot_3d_skeleton_sequence,
)
from humanproc_utils.pose_estimation_3d.mediapipe import (
    extract_3d_human_keypoints,
    extract_3d_human_keypoints_holistic,
)
from humanproc_utils.pose_estimation_3d.mediapipe import (
    load_pose_data,
    load_yaml_as_config,
    prepare_keypoints,
    save_keypoints,
)


# ------------------------------------------------------------------------------------------------------
def lists_of_dicts_to_one_list(lists):
    # Filter out empty lists
    lists = [lst for lst in lists if len(lst) > 0]
    return [{k: v for d in dicts for k, v in d.items()} for dicts in zip(*lists)]


# ------------------------------------------------------------------------------------------------------
def get_poses_main(load_json_path=None, add_post_processing=True):
    cfg = load_yaml_as_config("configs/pose_estimation/mediapipe.yaml")
    cfg_pose = cfg.pose_estimation

    if load_json_path is None:
        if cfg_pose.method == "mediapipe-holistic":
            keypoints_3d = extract_3d_human_keypoints_holistic(cfg_pose)
        elif cfg_pose.method == "mediapipe-separated":
            keypoints_3d = extract_3d_human_keypoints(cfg_pose)

        if cfg_pose.save_data:
            save_keypoints(keypoints_3d, cfg_pose.output_folder, name="mp_poses")

        if cfg_pose.show_3d:
            plot_3d_skeleton_sequence(keypoints_3d, add_hands=True)
    else:
        keypoints_3d = load_pose_data(load_json_path)

    if add_post_processing:
        (
            pose_3d_sequence_ready,
            left_hand_3d_sequence_ready,
            right_hand_3d_sequence_ready,
            face_mesh_3d_sequence_ready,
            pose_frames_inds,
            lh_frames_inds,
            rh_frames_inds,
            fm_frames_inds,
        ) = prepare_keypoints(keypoints_3d, scale_factor=0.5, is_smooth=False)

        keypoints_3d = {
            "poses": pose_3d_sequence_ready,
            "left_hand": left_hand_3d_sequence_ready,
            "right_hand": right_hand_3d_sequence_ready,
            "face": face_mesh_3d_sequence_ready,
            "pose_frames": pose_frames_inds,
            "lh_frames": lh_frames_inds,
            "rh_frames": rh_frames_inds,
            "fm_frames": fm_frames_inds,
        }

        if cfg_pose.save_data:
            save_numpy_keypoints(
                keypoints_3d, cfg_pose.output_folder, name="mp_poses_numpy"
            )
        if cfg_pose.show_3d:
            visualize_3d_skeleton_numpy(
                pose_3d_sequence_ready,
                left_hand_3d_sequence_ready,
                right_hand_3d_sequence_ready,
                face_mesh_3d_sequence_ready,
                lh_frames_inds,
                rh_frames_inds,
                fm_frames_inds,
            )


# ------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    get_poses_main(load_json_path=None, add_post_processing=False)
