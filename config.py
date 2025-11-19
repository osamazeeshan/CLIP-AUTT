import os
# import argparse

ROOT_DIR = os.environ["HOME"]

ROOT_DIR_LOCAL = '/projets/AS08960' 
ROOT_DIR_LOC_PRJ = '/projets/AS08960'

CURRENT_DIR = os.path.abspath(os.getcwd())

CURRENT_LOCAL_DIR = ROOT_DIR_LOCAL + '/Biovid'
CURRENT_HOME_DIR = ROOT_DIR + '/Biovid'

DATASET_FOLDER = "/datasets"

WEIGHTS_FOLDER = "WeightFiles"

_BIOVID_DATASET_LOCAL_PATH = ROOT_DIR_LOC_PRJ + DATASET_FOLDER      # Path for Biovid dataset local server

# 1. My local 
_DATASET_LOCAL_PATH = ROOT_DIR_LOCAL + DATASET_FOLDER
# 1. My home
_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER

DATASET_PATH = _FER_DATASET_PATH


#------------ Biovid Datasets
BIOVID = 'Biovid'
BIOVID_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid'
BIOVID_SOURCE_DATASET_PATH = DATASET_PATH + '/Biovid/sub_classes_tpt'
BIOVID_ALL_CAT_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img'
BIOVID_TAR_SUB_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img/081609_w_40/'
BIOVID_VIDEO_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/labels.txt'

BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_classes_img'
BIOVID_RED_SUBS_FOLDER = 'sub_red_classes_img'


#------------ Stresst Dataset

STRESS = 'StressID'
STRESS_PATH = _FER_DATASET_PATH + '/StressID'
STRESS_ALL_SUBS_PATH = _FER_DATASET_PATH + '/StressID/sub_images'

# -- all labels
STRESS_ALL_LABEL_PATH = _FER_DATASET_PATH + '/StressID/all_sub_labels.txt'
STRESS_ID_LABEL_VISUAL_PATH = _FER_DATASET_PATH + '/StressID/sub_N_source_classes.txt'
STRESS_SUBID_TO_SUBNAME_MAPPING = _FER_DATASET_PATH + '/StressID/sub_mapping_N_classes.txt'
STRESS_ID_LABEL_PHYSIO_PATH = _FER_DATASET_PATH + '/StressID/sub_N_source_class_physio.txt'

# -- all classes categorical for CAS class-aware
STRESS_ALL_CAT_SUBS_PATH = _FER_DATASET_PATH + '/StressID/sub_classes'


# ----------------- BAH ----------
BAH = 'BAH_DB'

PRETRAINED_BAH_CONFIG_FOLDER = CURRENT_HOME_DIR +'/config_files'
BAH_DATASET_PATH = _BIOVID_DATASET_LOCAL_PATH + '/BAH_DB'
BAH_TARGET_SPLIT_FOLDER = CURRENT_HOME_DIR + '/WeightFiles/target_split'
BAH_DATASET_FRAMES_PATH = BAH_DATASET_PATH + '/cropped-aligned-faces'
BAH_PATH_TRAIN = _BIOVID_DATASET_LOCAL_PATH + '/BAH_DB/split-frames/train.txt'
BAH_PATH_TRAIN_RB = DATASET_PATH + '/BAH_DB/split-frames/train_rb_20k.txt'

BAH_PATH_VAL = DATASET_PATH + '/BAH_DB/split-frames/val.txt'
BAH_PATH_TEST = DATASET_PATH + '/BAH_DB/split-frames/test.txt'

# ------------------------------------------

WEIGHTS_FOLDER = "WeightFiles"
BIOVID_N_SRC_WEIGHT_FILE = 'WeightFiles/lab_srcs78_cl77_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
ALL_SOURCES_FOLDER = "AllSources"

BIOVID_TRAIN_IMG_SRC_WEIGHTS = CURRENT_DIR + '/WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only_load.pt'
BIOVID_TRAIN_BIO_SRC_WEIGHTS = CURRENT_DIR + "/WeightFiles/bio_lab_srcs78_w45_m36_w60_m61_w23_w24_m62_w26_m40_m42_____only.pth"
MODEL_FUS_PATH = CURRENT_DIR + "/lab_srcs78_w45_m36_w60_m61_w23_w24_m62_w26_m40_m42_____only_fus_load.pt"

COMET_API_KEY = "API_KEY"
COMET_WORKSPACE = "WORKSPACE_NAME"
COMET_LOG_CODE = True
COMET_DISABLED = False
COMET_PROJECT_NAME = "PROJECT_NAME"

# -------------------
