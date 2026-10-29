"""Configuration settings, hyperparameters, and file paths."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RAW_DATA_PATH = DATA_DIR / "atmos41_raw.csv"

DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/pedroccpimenta/D4fa/master/HistoriMeteo/CVerdevarios.csv"
)

# Randomness
RANDOM_SEED = 42

# Time series parameters
SEQUENCE_LENGTH = 24
DAY_RADIATION_THRESHOLD = 10.0  # W/m^2 threshold for day vs night
TARGET_FEATURE = "temp_c"
FEATURES = ["temp_c", "pressure_kpa", "humidity_pct", "solar_rad_wm2"]

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.20
