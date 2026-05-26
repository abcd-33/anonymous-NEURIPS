SPHERE_MODEL = "custom"
DATA_DIR = "../sphere_encoder/workspace/experiments/" + SPHERE_MODEL + "/encoding/"

# DATA_DIR = "sphere_encoder/workspace/"

RAW_DATA_PATH = DATA_DIR + "encoded_dataset.npz"
PROC_DATA_PATH = DATA_DIR + "processed_dataset.npz"
OUTPUT_DATA_PATH = DATA_DIR + "output_encodings.npz"

SPHERE_DIMS = [64, 8]

SQUEEZE_DATA = False
SQUEEZE_ALPHA = 0.0
