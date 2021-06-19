NUM_VARIABLES = 3
ANGLE_SPLIT = 16
STRIDE = 8
MAX_ANOT_ANCHOR_POSITION_DISTANCE = 8
MAX_ANOT_ANCHOR_ANGLE_DISTANCE = (360.0/ANGLE_SPLIT) / 1.8
MAX_CORRECTABLE_DISTANCE = 5 * MAX_ANOT_ANCHOR_POSITION_DISTANCE
NAME, X, Y, ALPHA, LABEL, TRUTH, SCORE = 0, 1, 2, 3, 4, 5, 5

DAMPENING_PARAMETER = 0.0

ACC = {
    'LINE': (129, 178, 20),
    'CENTER': (248, 245, 241)
}

DEC = {
    'LINE': (255, 0, 92),
    'CENTER': (248, 245, 241)
}

RAW = {
    'LINE': (0, 0, 0),
    'CENTER': (0, 0, 0)
}
