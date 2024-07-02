from src.configs.constants import *
from src.data_utils.preprocess import (
    preprocess_mlm_data,
    preprocess_cls_data,
    preprocess_qa_data,
    preprocess_sts_data
)

def load_data(epoch:int = 0, seed:int = SEED):
    """By epoch, load data for one epoch."""
    
    # initialize the MLM data for epoch 0
    datasets_static_mlm = preprocess_mlm_data(epoch=epoch, seed = seed)

    # initialize the STS/retrieval data for epoch 0
    datasets_static_sts = preprocess_sts_data(epoch=epoch, seed = seed)
    
    # initialize the QA data for epoch 0
    datasets_static_qa = preprocess_qa_data(epoch=epoch, seed = seed)
    
    # initialize the CLS data for epoch 0
    datasets_static_cls = preprocess_cls_data(epoch=epoch, seed = seed)
    print('DONE')
    
    return {
        "mlm":datasets_static_mlm,
        "qa":datasets_static_qa,        
        "sts":datasets_static_sts,
        "cls":datasets_static_cls,
    }
    
    
