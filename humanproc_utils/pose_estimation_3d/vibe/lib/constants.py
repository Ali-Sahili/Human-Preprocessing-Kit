

#---------------------------------------------------------------------------------
LEFT_HAND_FINGERS = {
    # 'WRIST': ,
    'PINKY_TIP':  2673, 'PINKY_IP':   2630, 'PINKY_MCP':  2611, 'PINKY_CMC':  2600,
    'RING_TIP':   2556, 'RING_IP':    2517, 'RING_MCP':   2499, 'RING_CMC':   2480,
    'MIDDLE_TIP': 2445, 'MIDDLE_IP':  2406, 'MIDDLE_MCP': 2389, 'MIDDLE_CMC': 2362,
    'INDEX_TIP':  2319, 'INDEX_IP':	  2300, 'INDEX_MCP':  2204, 'INDEX_CMC':  2220,
    'THUMB_TIP':  2746, 'THUMB_IP':   2710, 'THUMB_MCP':  2704, 'THUMB_CMC':  2740,
}

#---------------------------------------------------------------------------------
RIGHT_HAND_FINGERS = {
    # 'WRIST': 5670,
    'PINKY_TIP':  6133, 'PINKY_IP':   6097, 'PINKY_MCP':  6051, 'PINKY_CMC':  5655,
    'RING_TIP':   6016, 'RING_IP':    5980, 'RING_MCP':   5960, 'RING_CMC':   5752,
    'MIDDLE_TIP': 5905, 'MIDDLE_IP':  5867, 'MIDDLE_MCP': 5850, 'MIDDLE_CMC': 5675,
    'INDEX_TIP':  5782, 'INDEX_IP':	  5757, 'INDEX_MCP':  5529, 'INDEX_CMC':  5735,
    'THUMB_TIP':  6191, 'THUMB_IP':   6169, 'THUMB_MCP':  5692, 'THUMB_CMC':  5687,
}

#---------------------------------------------------------------------------------
HAND_EDGES = [
    ('PINKY_TIP' , 'PINKY_IP'), ('PINKY_IP' , 'PINKY_MCP'), ('PINKY_MCP' , 'PINKY_CMC'), 
    ('PINKY_CMC', 'WRIST'),
    ('RING_TIP'  , 'RING_IP'),  ('RING_IP'  , 'RING_MCP'),  ('RING_MCP'  , 'RING_CMC'), 
    ('RING_CMC', 'WRIST'),
    ('MIDDLE_TIP', 'MIDDLE_IP'),('MIDDLE_IP', 'MIDDLE_MCP'),('MIDDLE_MCP', 'MIDDLE_CMC'), 
    ('MIDDLE_CMC', 'WRIST'),
    ('INDEX_TIP' , 'INDEX_IP'), ('INDEX_IP' , 'INDEX_MCP'), ('INDEX_MCP' , 'INDEX_CMC'),
    ('INDEX_CMC', 'WRIST'),
    ('THUMB_TIP' , 'THUMB_IP'), ('THUMB_IP' , 'THUMB_MCP'), ('THUMB_MCP' , 'THUMB_CMC'),
    ('THUMB_CMC', 'WRIST'),
]

#---------------------------------------------------------------------------------
FINGERS_TIP_INDICES = {
    'rthumb': 6191, 'rindex': 5782, 'rmiddle': 5905, 'rring': 6016, 'rpinky': 6133,
    'lthumb': 2746, 'lindex': 2319, 'lmiddle': 2445, 'lring': 2556, 'lpinky': 2673,
}

#---------------------------------------------------------------------------------
JOINTS_INDICES = {
    # EYES
    "RIGHT_EYE":      15, "RIGHT_EYE_2":    46,
    "LEFT_EYE":       16, "LEFT_EYE_2":     45,
    "RIGHT_EAR":      17, "RIGHT_EAR_2":    48,
    "LEFT_EAR":       18, "LEFT_EAR_2":     47,

    # HEAD
    "NOSE":            0, "NOSE_2":         44,
    "NECK":            1, "FRONT_NECK":     37,
    "TOP_HEAD":       38, "MOUTH":          42, 
    "MID_HEAD":       43,

    # RIGHT ARM
    "RIGHT_SHOULDER": 2, "RIGHT_SHOULDER_2": 33,
    "RIGHT_ELBOW":    3, "RIGHT_ELBOW_2":    31,
    "RIGHT_WRIST":    4, "RIGHT_WRIST_2":    32,

    # LEFT ARM
    "LEFT_SHOULDER":  5, "LEFT_SHOULDER_2":  34,
    "LEFT_ELBOW":     6, "LEFT_ELBOW_2":     35,
    "LEFT_WRIST":     7, "LEFT_WRIST_2":     36,

    # TORSO
    "MID_HIP":         8, "BEHIND_MID_HIP": 39,
    "IN_SHOULDERS":   40, "MID_BODY":       41,

    # RIGHT FOOT
    "RIGHT_LOWER_HIP":  9, "RIGHT_UPPER_HIP": 27,
    "RIGHT_KNEE":      26, "RIGHT_KNEE_2":    14,
    "RIGHT_ANKLE":     25, "RIGHT_ANKLE_2":   13,
    "RIGHT_BIG_TOE":   22, "RIGHT_SMALL_TOE": 23, 
    "RIGHT_HEEL":      24,
     
    # LEFT FOOT
    "LEFT_LOWER_HIP":  12, "LEFT_UPPER_HIP":  28,
    "LEFT_KNEE":       29, "LEFT_KNEE_2":       10,
    "LEFT_ANKLE":      30, "LEFT_ANKLE_2":      11,
    "LEFT_BIG_TOE":    19, "LEFT_SMALL_TOE":  20,
    "LEFT_HEEL":       21,
}

#---------------------------------------------------------------------------------
SKELETON_EDGES = [
    (0,42), (0,43), (43,38), (15,0), (16,0), (15,17), (16,18),
    (37,42),
    (40,41), (40, 5), (40, 2), (40, 37), (1,37),
    (2,3), (3,4),     # RIGHT ARM
    (5,6), (6,7),     # LEFT ARM
    (12,13), (12,28), (29,30), (30,21), (19,21), (20,21), (19,20), # LEFT FOOT
    (22,23), (22,24), (23,24), (24,25), (25,26), (26,9), (9,27),  # RIGHT FOOT
    (27,39), (28,39),
    (8,41), (8,39)
]