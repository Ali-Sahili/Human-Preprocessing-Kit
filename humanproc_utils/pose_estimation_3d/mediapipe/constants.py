
import mediapipe as mp

#------------------------------------------------------------------------------------------------------
POSE_LANDMARKS_NAMES = [landmark.name for landmark in mp.solutions.pose.PoseLandmark]
HAND_LANDMARKS_NAMES = [landmark.name for landmark in mp.solutions.hands.HandLandmark]
FULL_FACEMESH_LANDMARKS_NAMES = { **{f"landmark_{i}": i for i in range(0, 468)} }
FULL_FACEMESH_LANDMARKS_NAMES_LIST = [f"landmark_{i}" for i in range(0, 468)]

# Root between left hip and right hip
ROOT_IN_BETWEEN = [POSE_LANDMARKS_NAMES[23], POSE_LANDMARKS_NAMES[24]]

# Keypoints common between faces, hands and poses - to be removed from poses
# REDUNDANCY_TO_REMOVE = [
#         POSE_LANDMARKS_NAMES.index('NOSE'), 
#         POSE_LANDMARKS_NAMES.index('LEFT_EYE_INNER'), POSE_LANDMARKS_NAMES.index('LEFT_EYE'), 
#         POSE_LANDMARKS_NAMES.index('LEFT_EYE_OUTER'), POSE_LANDMARKS_NAMES.index('RIGHT_EYE_INNER'), 
#         POSE_LANDMARKS_NAMES.index('RIGHT_EYE'), POSE_LANDMARKS_NAMES.index('RIGHT_EYE_OUTER'), 
#         POSE_LANDMARKS_NAMES.index('LEFT_EAR'), POSE_LANDMARKS_NAMES.index('RIGHT_EAR'), 
#         POSE_LANDMARKS_NAMES.index('MOUTH_LEFT'), POSE_LANDMARKS_NAMES.index('MOUTH_RIGHT'),
#         POSE_LANDMARKS_NAMES.index("LEFT_PINKY"), POSE_LANDMARKS_NAMES.index("RIGHT_PINKY"),
#         POSE_LANDMARKS_NAMES.index("LEFT_INDEX"), POSE_LANDMARKS_NAMES.index("RIGHT_INDEX"),
#         POSE_LANDMARKS_NAMES.index("LEFT_THUMB"), POSE_LANDMARKS_NAMES.index("RIGHT_THUMB")
#     ]

REDUNDANCY_FACE_MESH_TO_REMOVE = [
        POSE_LANDMARKS_NAMES.index('NOSE'), 
        POSE_LANDMARKS_NAMES.index('LEFT_EYE_INNER'), POSE_LANDMARKS_NAMES.index('LEFT_EYE'), 
        POSE_LANDMARKS_NAMES.index('LEFT_EYE_OUTER'), POSE_LANDMARKS_NAMES.index('RIGHT_EYE_INNER'), 
        POSE_LANDMARKS_NAMES.index('RIGHT_EYE'), POSE_LANDMARKS_NAMES.index('RIGHT_EYE_OUTER'), 
        POSE_LANDMARKS_NAMES.index('LEFT_EAR'), POSE_LANDMARKS_NAMES.index('RIGHT_EAR'), 
        POSE_LANDMARKS_NAMES.index('MOUTH_LEFT'), POSE_LANDMARKS_NAMES.index('MOUTH_RIGHT'),
    ]

REDUNDANCY_LEFT_HAND_TO_REMOVE = [ POSE_LANDMARKS_NAMES.index("LEFT_PINKY"), 
                                   POSE_LANDMARKS_NAMES.index("LEFT_INDEX"),
                                   POSE_LANDMARKS_NAMES.index("LEFT_THUMB")
                            ]

REDUNDANCY_RIGHT_HAND_TO_REMOVE = [ POSE_LANDMARKS_NAMES.index("RIGHT_PINKY"),
                                    POSE_LANDMARKS_NAMES.index("RIGHT_INDEX"),
                                    POSE_LANDMARKS_NAMES.index("RIGHT_THUMB")
                            ]

#------------------------------------------------------------------------------------------------------
# 
head_body_indices = POSE_LANDMARKS_NAMES[0:11]
upper_body_indices = POSE_LANDMARKS_NAMES[11:23]
lower_body_indices = POSE_LANDMARKS_NAMES[23:]

#------------------------------------------------------------------------------------------------------
# MediaPipe Pose landmark connections (pairs of indexes)
# POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),                             # Head to left shoulder
    (0, 4), (4, 5), (5, 6), (6, 8),                             # Head to right shoulder
    (9, 10),                                                    # Shoulders
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),           # Right arm
    (11, 23), (12, 24), (23, 24),                               # Torso
    (23, 25), (24, 26), (25, 27), (27, 29), (29, 31), (27, 31),         # Left leg
    (26, 28), (28, 30), (30, 32), (28, 32)                                # Right leg
]

#------------------------------------------------------------------------------------------------------
# Define hand landmark connections for drawing (same for both hands)
# HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

#------------------------------------------------------------------------------------------------------

FACEMESH_LANDMARKS_NAMES = {
    "nose_tip": 1,
    "nose_bridge": 6,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "left_eye_top": 158,
    "left_eye_bottom": 144,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "right_eye_top": 386,
    "right_eye_bottom": 373,
    "left_eyebrow_inner": 52,
    "left_eyebrow_outer": 55,
    "right_eyebrow_inner": 282,
    "right_eyebrow_outer": 285,
    "mouth_left_corner": 61,
    "mouth_right_corner": 291,
    "upper_lip_top": 13,
    "upper_lip_bottom": 0,
    "lower_lip_top": 14,
    "chin": 152,
    "jaw_left": 234,
    "jaw_right": 454,
}
