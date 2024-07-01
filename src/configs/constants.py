SEED = 1982
MAX_SEQ_LENGTH=512 # max number of tokens
MIN_SEQ_LENGTH=48 # min number of tokens to qualify as an example
MAX_CHUNK_SIZE = 6 # splitting long-docs into X chunks of MAX_SEQ_LENGTH
CHAR_PER_WORD = 6.36 # fudge factor for number of characters per word
NEGATIVE_CORPUS_SIZE = 1000 # size of negative corpus for generating soft negatives for triplets
TRAINING_CORPUS_SIZE_PER_EPOCH = 5000
VAL_CORPUS_SIZE = 500
PATH_CACHE_MLM_VAL = "/tmp/cache_val_mlm.pkl"
PATH_CACHE_MLM_TRAIN = "/tmp/cache_train_mlm_%03g.pkl"

print(f'WARNING: NEGATIVE_CORPUS_SIZE is {NEGATIVE_CORPUS_SIZE} should be 40000')
print(f'WARNING: TRAINING_CORPUS_SIZE_PER_EPOCH is {TRAINING_CORPUS_SIZE_PER_EPOCH} should be 20000')
print(f'WARNING: VAL_CORPUS_SIZE is {VAL_CORPUS_SIZE} should be 5000')


