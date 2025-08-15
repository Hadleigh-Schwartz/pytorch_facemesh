"""
Eye landmark ids, per https://github.com/google-ai-edge/mediapipe/blob/ab1de4fced96c18b79eda4ab407b04eb08301ea4/mediapipe/graphs/iris_tracking/calculators/update_face_landmarks_calculator.cc#L33
These are in the order that the 71 eye contour landmarks returned by the separate iris landmarker model are returned in.
The above code and description here https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md
indicate that in the official FaceLandmarker (478 landmarks - 468 facial and 10 extra iris) release, the eye contour landmarks obtained are actually a refined version of the original 71 landmarks
returned by the 468-point facial landmarker, obtained via the iris landmarking run under the hood after getting an initial guess from the 468-point model.
"""

eye_landmark_ids = [   # Left eye
    # eye lower contour
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    # eye upper contour (excluding corners)
    246, #9
    161,
    160,
    159,
    158,
    157,
    173,
    # halo x2 lower contour
    130,#16
    25,
    110,
    24,
    23,
    22,
    26,
    112,
    243,
    # halo x2 upper contour (excluding corners)
    247, #25
    30,
    29,
    27,
    28,
    56,
    190, 
    # halo x3 lower contour
    226, # 32
    31,
    228,
    229,
    230,
    231,
    232,
    233,
    244,
    # halo x3 upper contour (excluding corners)
    113, # 41
    225,
    224,
    223,
    222,
    221,
    189,
    # halo x4 upper contour (no lower because of mesh structure)
    # or eyebrow inner contour
    35, #48
    124,
    46,
    53,
    52,
    65,
    # halo x5 lower contour
    143,#54
    111,
    117,
    118,
    119,
    120,
    121,
    128,
    245,
    # halo x5 upper contour (excluding corners)
    # or eyebrow outer contour
    156, # 63
    70,
    63,
    105,
    66,
    107,
    55,
    193,

    # # Right eye
    # eye lower contour
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    # eye upper contour (excluding corners)
    466, # 9
    388,
    387,
    386,
    385,
    384,
    398,
    # halo x2 lower contour
    359,
    255,
    339,
    254,
    253,
    252,
    256,
    341,
    463,
    # halo x2 upper contour (excluding corners)
    467, #32
    260,
    259,
    257,
    258,
    286,
    414,
    # halo x3 lower contour
    446,
    261,
    448,
    449,
    450,
    451,
    452,
    453,
    464,
    # halo x3 upper contour (excluding corners)
    342,
    445,
    444,
    443,
    442,
    441,
    413,
    # halo x4 upper contour (no lower because of mesh structure)
    # or eyebrow inner contour
    265,
    353,
    276,
    283,
    282,
    295,
    # halo x5 lower contour
    372,
    340,
    346,
    347,
    348,
    349,
    350,
    357,
    465,
    # halo x5 upper contour (excluding corners)
    # or eyebrow outer contour
    383,
    300,
    293,
    334,
    296,
    336,
    285,
    417
]