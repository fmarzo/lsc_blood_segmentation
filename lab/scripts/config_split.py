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
TRAIN_VIDEO_ID = ["pig1", "pig3", "pig4", "pig6", "pig9", "pig11"]
VAL_VIDEO_ID = ["pig2", "pig7"]
TEST_VIDEO_ID = ["pig5", "pig10"]
