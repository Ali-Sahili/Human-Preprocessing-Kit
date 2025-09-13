import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from humanproc_utils.pose_estimation_3d.vibe.lib.utils.utils import read_pkl_file


# ---------------------------------------------------------------------------------
def plot_3d_pose_sequence(
    joints3d,
    verts,
    add_mesh=True,
    frame_step=1,
    remove_axis=False,
    remove_axis_labels=False,
    nb_frames_to_visualize=0,
):
    """
    Visualize the 3D pose sequence and vertices from VIBE output.

    Parameters:
    - joints3d: numpy array, shape (frames, joints, 3) representing 3D joints positions.
    - verts: numpy array, shape (frames, vertices, 3) representing 3D vertices positions.
    - frame_step: int, step for sampling frames for visualization.
    """
    num_frames, num_joints, _ = joints3d.shape

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=-90, azim=-90)
    # # ax.view_init(elev=-10, azim=-133)
    # ax.view_init(elev=21, azim=-100)

    if nb_frames_to_visualize > 0 and nb_frames_to_visualize <= num_frames:
        num_frames = nb_frames_to_visualize

    # Loop through selected frames and plot 3D skeleton and vertices
    for i in range(0, num_frames, frame_step):
        ax.cla()
        ax.set_title(f"Frame {i + 1}/{num_frames}")

        # Plot the skeleton joints
        joint_coords = joints3d[i]
        ax.scatter(
            joint_coords[:, 0],
            joint_coords[:, 1],
            joint_coords[:, 2],
            color="blue",
            s=10,
            label="Joints",
        )

        # Plot the vertices as a mesh
        if add_mesh:
            verts_coords = verts[i]
            ax.scatter(
                verts_coords[:, 0],
                verts_coords[:, 1],
                verts_coords[:, 2],
                color="orange",
                s=1,
                alpha=0.5,
                label="Vertices",
            )

        # Set limits and labels for better visibility
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if remove_axis:
            ax.set_axis_off()
        if remove_axis_labels:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        plt.pause(0.1)

    plt.show()


# ---------------------------------------------------------------------------------
def temporal_smoothing(sequence, alpha=0.1):
    """
    Apply Exponential Moving Average (EMA) for temporal smoothing of a sequence.

    Parameters:
        sequence (list of numpy arrays): A sequence of 3D positions (vertices or joints).
        alpha (float): Smoothing factor between 0 and 1. Higher means less smoothing.

    Returns:
        smoothed_sequence (list of numpy arrays): Smoothed sequence.
    """
    smoothed_sequence = []
    # Initialize with the first frame
    smoothed_frame = sequence[0]
    smoothed_sequence.append(smoothed_frame)

    # Apply EMA for each frame
    for i in range(1, sequence.shape[0]):
        smoothed_frame = alpha * sequence[i] + (1 - alpha) * smoothed_frame
        smoothed_sequence.append(smoothed_frame)

    return smoothed_sequence


# ---------------------------------------------------------------------------------
def scale_normalization(vertices_seq, joints_seq, target_height=0.8):

    pelvis_idx, head_idx = 8, 38
    scaled_vertices_seq, scaled_joints_seq = [], []

    # Calculate the average height across frames based on pelvis to head distance
    heights = []
    for joints in joints_seq:
        # Estimate height as the vertical distance between pelvis and head
        height = np.linalg.norm(joints[head_idx] - joints[pelvis_idx])
        heights.append(height)

    avg_height = np.mean(heights)
    scale_factor = target_height / avg_height

    # Scale vertices and joints in each frame
    for vertices, joints in zip(vertices_seq, joints_seq):
        scaled_vertices_seq.append(vertices * scale_factor)
        scaled_joints_seq.append(joints * scale_factor)

    return scaled_vertices_seq, scaled_joints_seq


# ---------------------------------------------------------------------------------
input_path = "outputs/tcmr/masked_running_man_2sec/tcmr_output.pkl"
vibe_output = read_pkl_file(input_path)
print("NUmber of detected persons: ", len(vibe_output))

if len(vibe_output) > 0:
    for id in vibe_output.keys():
        print(f"Person ID: {id}")
        vertices_seq = vibe_output[id]["verts"]
        joints3d_seq = vibe_output[id]["joints3d"]

        # plot_3d_pose_sequence(joints3d_seq, vertices_seq, frame_step=1, nb_frames_to_visualize=0)

        # Step 1: Apply temporal smoothing
        smoothed_vertices_seq = temporal_smoothing(vertices_seq, alpha=0.1)
        smoothed_joints_seq = temporal_smoothing(joints3d_seq, alpha=0.1)

        # Step 2: Apply scale normalization
        target_height = 0.8  # Assume average human height of 1.7 meters
        scaled_vertices_seq, scaled_joints_seq = scale_normalization(
            smoothed_vertices_seq, smoothed_joints_seq, target_height=target_height
        )
        scaled_vertices_seq, scaled_joints_seq = np.array(
            scaled_vertices_seq
        ), np.array(scaled_joints_seq)

        plot_3d_pose_sequence(
            scaled_joints_seq,
            scaled_vertices_seq,
            frame_step=1,
            nb_frames_to_visualize=0,
        )
