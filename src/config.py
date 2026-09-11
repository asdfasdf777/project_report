from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

FMA_AUDIO_DIR = RAW_DIR / "fma_small"
FMA_METADATA_DIR = RAW_DIR / "fma_metadata"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

MAX_TRACKS = 1000

NUM_LABELS = 20

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# --------------------------------------------------
# Audio
# --------------------------------------------------

SAMPLE_RATE = 22050

SEGMENT_SECONDS = 5

N_MELS = 128

N_MFCC = 13

N_CHROMA = 12


# --------------------------------------------------
# BERT
# --------------------------------------------------

BERT_MODEL_NAME = "bert-base-uncased"

MAX_TEXT_LENGTH = 128


# --------------------------------------------------
# Training
# --------------------------------------------------

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 5

RANDOM_SEED = 42


# --------------------------------------------------
# Graph
# --------------------------------------------------

GRAPH_SIMILARITY_THRESHOLD = 0.8

GNN_HIDDEN_DIM = 64

FUSION_HIDDEN_DIM = 128