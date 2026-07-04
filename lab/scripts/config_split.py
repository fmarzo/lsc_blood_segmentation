import os
# getting the actual path and pointing to the final one
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))

#configuration contants
DATASET_ROOT = '/work/cvcs2026/latent_space_cowboys/datasets/HemoSet'
OUT_DIR_SPLIT = os.path.join(PROJECT_ROOT, 'splits')
CSV_FILE_NAME = os.path.join(OUT_DIR_SPLIT, 'full_labeled_dataset.csv')
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
PIG8    = "pig8"
PIG9    = "pig9"
PIG10   = "pig10" 
PIG11   = "pig11"

TRAIN_VIDEO_ID = [PIG1, PIG3, PIG4, PIG6, PIG7, PIG8]
VAL_VIDEO_ID =   [PIG2, PIG9, PIG11]
TEST_VIDEO_ID =  [PIG5, PIG10]
