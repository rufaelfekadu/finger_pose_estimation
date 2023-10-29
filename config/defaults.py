from yacs.config import CfgNode as CN

_C = CN()

# ------------------------------------
# DATA
# ------------------------------------

_C.DATA = CN()
_C.DATA.PATH = "./dataset"
_C.DATA.SEGMENT_LENGTH = 5

_C.DATA.EMG = CN()
_C.DATA.EMG.SAMPLING_RATE = 250
_C.DATA.EMG.NUM_CHANNELS = 16
_C.DATA.EMG.FEATURE_EXTRACTOR = "RMS"
_C.DATA.EMG.WINDOW_SIZE = 100
_C.DATA.EMG.WINDOW_STRIDE = 50
_C.DATA.EMG.HIGH_PASS = 400
_C.DATA.EMG.LOW_PASS = 10
_C.DATA.EMG.NOTCH = 50
_C.DATA.EMG.NORMALIZATION = "max"

_C.DATA.VIDEO = CN()
_C.DATA.VIDEO.SAMPLING_RATE = 30
_C.DATA.VIDEO.NUM_CHANNELS = 3

_C.DATA.MANUS = CN()
_C.DATA.MANUS.SAMPLING_RATE = 250


# ------------------------------------
# MODEL
# ------------------------------------

# ------------------------------------
# OUTPUT
# ------------------------------------

# ------------------------------------
# Solver
# ------------------------------------

_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
