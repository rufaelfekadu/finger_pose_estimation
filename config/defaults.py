from yacs.config import CfgNode as CN

_C = CN()
_C.DEBUG = True

# ------------------------------------
# DATA
# ------------------------------------

_C.DATA = CN()
_C.DATA.PATH = "finger_pose_estimation/dataset/data_2023-10-02 14-59-55-627.edf"
_C.DATA.LABEL_PATH = "finger_pose_estimation/dataset/label_2023-10-02_15-24-12_YH_lab_R.csv"
_C.DATA.SEGMENT_LENGTH = 150

_C.DATA.EMG = CN()
_C.DATA.EMG.SAMPLING_RATE = 150
_C.DATA.EMG.NUM_CHANNELS = 16
_C.DATA.EMG.FEATURE_EXTRACTOR = "RMS"
_C.DATA.EMG.WINDOW_SIZE = 100
_C.DATA.EMG.WINDOW_STRIDE = 50
_C.DATA.EMG.HIGH_FREQ = 400
_C.DATA.EMG.LOW_FREQ = 10
_C.DATA.EMG.NOTCH = 50
_C.DATA.EMG.NORMALIZATION = "max"
_C.DATA.EMG.NOTCH_FREQ = 50
_C.DATA.EMG.BUFF_LEN = 0

_C.DATA.VIDEO = CN()
_C.DATA.VIDEO.SAMPLING_RATE = 30
_C.DATA.VIDEO.NUM_CHANNELS = 3

_C.DATA.MANUS = CN()
_C.DATA.MANUS.SAMPLING_RATE = 250
_C.DATA.MANUS.NUM_JOINTS = 15
_C.DATA.MANUS.KEY_POINTS = []


# ------------------------------------
# MODEL
# ------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "neuropose"

# ------------------------------------
# OUTPUT
# ------------------------------------

# ------------------------------------
# Solver
# ------------------------------------

_C.SOLVER = CN()

_C.SOLVER.METRIC = "mse"
_C.SOLVER.PATIENCE = 5

# Optimizer
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.CHECKPOINT_PERIOD = 10

# Miscellaneus
_C.SOLVER.LOG_DIR = "./outputs"
_C.SOLVER.DEVICE = "cuda"
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.PIN_MEMORY = True
_C.SOLVER.SEED = 42
_C.SOLVER.PRETRAINED_PATH = "model.pth"
