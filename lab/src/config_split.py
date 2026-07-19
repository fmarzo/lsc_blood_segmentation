import os
# getting the actual path and pointing to the final one
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
WORK_ROOT = '/work/cvcs2026/latent_space_cowboys/'

#configuration contants
DATASET_ROOT = os.path.join(WORK_ROOT,'datasets/HemoSet')
OUT_DIR_SPLIT = os.path.join(PROJECT_ROOT, 'splits')
CSV_FILE_PATH = os.path.join(OUT_DIR_SPLIT, 'full_labeled_dataset.csv')
CSV_TRAIN_PATH = os.path.join(OUT_DIR_SPLIT, 'train.csv')
CSV_VALID_PATH = os.path.join(OUT_DIR_SPLIT, 'val.csv')
CSV_TEST_PATH = os.path.join(OUT_DIR_SPLIT, 'test.csv')

BLEEDING_DATASET_ROOT = os.path.join(WORK_ROOT,'datasets', 'bleeding_segmentation_v1p0',)
BLEEDING_IMAGES_DIR = os.path.join(BLEEDING_DATASET_ROOT, 'images',)
BLEEDING_MASKS_DIR = os.path.join(BLEEDING_DATASET_ROOT, 'masks',)
OUT_DIR_SPLIT_V1P0 = os.path.join(PROJECT_ROOT,'splits_v1p0',)
CSV_FILE_PATH_V1P0 = os.path.join(OUT_DIR_SPLIT_V1P0,'full_labeled_dataset.csv',)
BLEEDING_IMG_EXT = '.jpg'
BLEEDING_MASK_EXT = '.png'
BLEEDING_VIDEO_ID = 'bleeding_segmentation_v1p0'
CSV_TRAIN_PATH_V1P0 = os.path.join(OUT_DIR_SPLIT_V1P0, 'train.csv')
CSV_VALID_PATH_V1P0 = os.path.join(OUT_DIR_SPLIT_V1P0, 'val.csv')
CSV_TEST_PATH_V1P0 = os.path.join(OUT_DIR_SPLIT_V1P0, 'test.csv')

DEFAULT_EPOCHS = 5
NUM_DATASET_FOLDERS = 11
IMG_EXT = '.png'
IMG_STRING = 'images'
LABEL_STRING = 'labels'
VIDEOID_STRING = 'video_id'

#dataset split constants

PIG1    = "pig1"
PIG2    = "pig2"
PIG3    = "pig3"
PIG4    = "pig4"
PIG5    = "pig5"
PIG6    = "pig6"
PIG7    = "pig7"
PIG9    = "pig9"
PIG10   = "pig10" 
PIG11   = "pig11"

# TRAIN_VIDEO_ID = [PIG1, PIG2, PIG3, PIG4, PIG5, PIG6, PIG7]
# VAL_VIDEO_ID =   [PIG9]
# TEST_VIDEO_ID =  [PIG10, PIG11]

#SPLIT OF BASELINE
TRAIN_VIDEO_ID = [PIG1, PIG2, PIG9, PIG10]
VAL_VIDEO_ID =   [PIG5, PIG7]
TEST_VIDEO_ID =  [PIG3, PIG6, PIG11]

SEGMENTATION_MODE = "binary"   # "binary" oppure "multiclass"
NUM_CLASSES = 1 if SEGMENTATION_MODE == "binary" else 2
BINARY_THRESHOLD = 0.5
MODEL_PRETRAINED_DIR = os.path.join(
    WORK_ROOT,
    "model_pretrained",
)

# EVALUATION MODEL PARAM FOR HEMOSET,, USED ALSO FOR ZERO SHOT ON RABBANI SET TEST (encoder = resnet18)
MODEL_TO_EVALUATE = "unet" # "unet_plus_plus" or "unet"
ENCODER_NAME = "resnet18"

# EVALUATION MODEL PARAM FOR RABBANI, USED ALSO FOR ZERO SHOT ON HEMOSET SET TEST
RABBANI_EVALUATION_MODEL = "unet_plus_plus" # "deeplabv3plus" or "unet_plus_plus" 
# The encoder is always a resnet18