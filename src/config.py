BATCH_SIZE = 64

EPOCHS = 2

WARMUP_STEPS = 100

EVAL_STEPS = 1000

LEARNING_RATE = 2e-5

# Sent tranformer type
SBERT_INIT = "all-MiniLM-L12-v2"

# where to save model and logs
SBERT_SAVE_PATH = "s3bert_" + SBERT_INIT + "/"

# Data path to AMR data set
DATA_PATH = "../amr_data_set/"

# where to save pre-processed data
PREPRO_PATH = "preprocessed/"

# two sentence input features names
FEATURES = ['input1', 'input2']

# metric score feature names
FEATURES += ['Concepts ', 'Frames ', 'Named Ent. ',
            'Negations ', 'Reentrancies ', 'SRL ', 'Smatch ', 'Unlabeled ',
            'max_indegree_sim', 'max_outdegree_sim', 'max_degree_sim',
            'root_sim', 'quant_sim', 'score_wlk', 'score_wwlk']

# number of metrics
N = 15

# Feature dimension
FEATURE_DIM = 16





