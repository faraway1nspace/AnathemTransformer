SEED = 1982
MAX_SEQ_LENGTH=512 # max number of tokens
MIN_SEQ_LENGTH=48 # min number of tokens to qualify as an example
MAX_CHUNK_SIZE = 6 # splitting long-docs into X chunks of MAX_SEQ_LENGTH
CHAR_PER_WORD = 6.36 # fudge factor for number of characters per word
TOKEN_FUDGE_FACTOR = 1.36 # ratio of tokens to words
NEGATIVE_CORPUS_SIZE = 40000 # size of negative corpus for generating soft negatives for triplets
TRAINING_CORPUS_SIZE_PER_EPOCH = 10000
VAL_CORPUS_SIZE = 1000 # size of corpus of validation sets, per task
TRIPLETS_N_NEGATIVES = 3 # number of negatives to retrieve for triplets
TRIPLETS_TOPK_NEGATIVES = 15 # for soft negatives, discard top K closest matches from retrievre
NEXTSENTENCE_MIN_N_SENTENCES = 18 # minimum number of sentences to extract next-sentence task
CACHE_DIR = "/content/drive/MyDrive/ScriptsPrograms/ml_anathem_transformer/cached_data" # where caches will be saved
PATH_CACHE_MLM_VAL = f"{CACHE_DIR}/cache_val_mlm.pkl"
PATH_CACHE_MLM_TRAIN = f"{CACHE_DIR}/cache_train_mlm_%03g.pkl"
PATH_CACHE_QA_VAL = f"{CACHE_DIR}/cache_val_qa.pkl"
PATH_CACHE_QA_TRAIN = f"{CACHE_DIR}/cache_train_qa_%03g.pkl"
PATH_CACHE_STS_VAL = f"{CACHE_DIR}/cache_val_sts.pkl"
PATH_CACHE_STS_TRAIN = f"{CACHE_DIR}/cache_train_sts_%03g.pkl"
PATH_CACHE_CLS_VAL = f"{CACHE_DIR}/cache_val_cls.pkl"
PATH_CACHE_CLS_TRAIN = f"{CACHE_DIR}/cache_train_cls_%03g.pkl"
PATH_CACHE_NEGATIVES = f'{CACHE_DIR}/negative_corpus_cache.pkl'
NEGATIVE_CORPUS_METHOD_STS = 'ann-tfidf'
NEGATIVE_CORPUS_METHOD_QA = 'ann-tfidf' #'bm25'
DISTILLATION_TEMPERATURE = 1.5

DIR_LOG = "/content/drive/MyDrive/ScriptsPrograms/ml_anathem_transformer/cached_data/"
DEFAULT_PROB_QA = 0.1

print(f'WARNING: NEGATIVE_CORPUS_SIZE is {NEGATIVE_CORPUS_SIZE} should be 40000')
print(f'WARNING: TRAINING_CORPUS_SIZE_PER_EPOCH is {TRAINING_CORPUS_SIZE_PER_EPOCH} should be 20000')
print(f'WARNING: VAL_CORPUS_SIZE is {VAL_CORPUS_SIZE} should be 5000')

