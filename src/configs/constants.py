SEED = 1982
MAX_SEQ_LENGTH=512 # max number of tokens
MIN_SEQ_LENGTH=48 # min number of tokens to qualify as an example
MAX_CHUNK_SIZE = 6 # splitting long-docs into X chunks of MAX_SEQ_LENGTH
CHAR_PER_WORD = 6.36 # fudge factor for number of characters per word
TOKEN_FUDGE_FACTOR = 1.36
NEGATIVE_CORPUS_SIZE = 1000 # size of negative corpus for generating soft negatives for triplets
TRAINING_CORPUS_SIZE_PER_EPOCH = 5000
VAL_CORPUS_SIZE = 500
CACHE_DIR = "/tmp" # where caches will be saved
PATH_CACHE_MLM_VAL = f"{CACHE_DIR}/cache_val_mlm.pkl"
PATH_CACHE_MLM_TRAIN = f"{CACHE_DIR}/cache_train_mlm_%03g.pkl"
PATH_CACHE_QA_VAL = f"{CACHE_DIR}/cache_val_qa.pkl"
PATH_CACHE_QA_TRAIN = f"{CACHE_DIR}/cache_train_qa_%03g.pkl"
PATH_CACHE_STS_VAL = f"{CACHE_DIR}/cache_val_sts.pkl"
PATH_CACHE_STS_TRAIN = f"{CACHE_DIR}/cache_train_sts_%03g.pkl"


DIR_LOG = "/tmp/"
DEFAULT_PROB_QA = 0.1

print(f'WARNING: NEGATIVE_CORPUS_SIZE is {NEGATIVE_CORPUS_SIZE} should be 40000')
print(f'WARNING: TRAINING_CORPUS_SIZE_PER_EPOCH is {TRAINING_CORPUS_SIZE_PER_EPOCH} should be 20000')
print(f'WARNING: VAL_CORPUS_SIZE is {VAL_CORPUS_SIZE} should be 5000')


