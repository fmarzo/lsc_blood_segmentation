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

TRAIN_VIDEO_ID = [PIG1, PIG2, PIG9, PIG10]
VAL_VIDEO_ID =   [PIG3, PIG5, PIG7]
TEST_VIDEO_ID =  [PIG4, PIG6, PIG11]

SEGMENTATION_MODE = "binary"   # "binary" oppure "multiclass"
NUM_CLASSES = 1 if SEGMENTATION_MODE == "binary" else 2
BINARY_THRESHOLD = 0.5
MODEL_PRETRAINED_DIR = os.path.join(
    WORK_ROOT,
    "model_pretrained",
)

MODEL_TO_EVALUATE = "unet" # or "unet"

ENCODER_NAME = "resnet18"