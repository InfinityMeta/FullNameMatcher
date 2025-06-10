import os
import faiss
from pathlib import Path


CURRENT_DIR = Path.cwd()

DATA_DIR = CURRENT_DIR / 'data'
MODELS_DIR = CURRENT_DIR / 'models'
LOGS_DIR = CURRENT_DIR / 'logs'

for dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(MODELS_DIR)

DATA_DIR = CURRENT_DIR / 'data'
LOGS_DIR = CURRENT_DIR / 'logs'

for dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / 'dataset.json'
PROCESSED_DATASET_PATH = DATA_DIR / 'processed_dataset.pkl'
VOCABULARY_PATH = DATA_DIR / 'vocabulary.pkl'
ANN_INDEX_PATH = MODELS_DIR / 'ann_index.index'
LOGS_FILE_PATH = LOGS_DIR / 'matcher.logs'

RERANKER_NAME = 'CloudlessSky/fullname_encoder_v1'
SEARCH_FIELD_NAME = 'Name'
EMBEDDINGS_DIM = 384

BIGRAMS_MATCHES_NUM = 30
RERANKER_THRESHOLD = 0.9

ANN_PARAMS = {
    'centroids_num_fraction': 0.5,
    'nprobe_num_fraction': 0.65,
    'index_name': 'IVF{centroids_num},Flat',
    'index_metric': faiss.METRIC_INNER_PRODUCT
}
