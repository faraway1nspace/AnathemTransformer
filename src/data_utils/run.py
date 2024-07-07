
from torch.utils.data import Dataset as TorchDataset
from typing import Dict
from src.configs.constants import *
from src.data_utils.preprocess import (
    preprocess_mlm_data,
    preprocess_cls_data,
    preprocess_qa_data,
    preprocess_sts_data,
)
from src.data_utils.dataloaders import make_torch_datasets, reformat_nextsentence_for_cls_task
# spacy.language.Language

def load_data(
        epoch:int = 0,
        seed:int = SEED
) -> Dict[str,Dict[str, TorchDataset]]:
    """By epoch, load all task data for one epoch, convert to torch.data.utils.Dataset."""
    
    # initialize the MLM data for epoch 0
    datasets_static_mlm, datasets_static_nextsent  = preprocess_mlm_data(
        epoch=epoch, seed = seed
    )

    # initialize the QA data for epoch 0
    datasets_static_qa = preprocess_qa_data(epoch=epoch, seed = seed)
    
    # initialize the STS/retrieval data for epoch 0
    datasets_static_sts = preprocess_sts_data(epoch=epoch, seed = seed)
    
    # initialize the CLS data for epoch 0
    datasets_static_cls = preprocess_cls_data(epoch=epoch, seed = seed)
    # DONE (re)loading the streaming data for epoch
    # Next, we make torch.data.util.Dataset for the tasks
    
    # make the torch dataset MLM
    tdata_mlm = make_torch_datasets(datasets_static_mlm)
    
    # make the torch dataset QA
    tdata_qa = make_torch_datasets(datasets_static_qa)
    
    # make the torch dataset STS
    tdata_sts = make_torch_datasets(datasets_static_sts)
    
    # make the torch dataset CLS
    tdata_cls = make_torch_datasets(datasets_static_cls)
    
    print('DONE loading all tasks and making torch Datasets')
    
    # merge the NextSentence Datsaet into the CLS task (i.e., predict next sentence or not)
    tdata_cls['train']._integrate_another_dataset(
        list_of_newdata = datasets_static_nextsent.train,
        function_to_reformatdata = reformat_nextsentence_for_cls_task,
        dataset_name = 'nextsentence',
    )
    tdata_cls['val']._integrate_another_dataset(
        list_of_newdata = datasets_static_nextsent.val,
        function_to_reformatdata = reformat_nextsentence_for_cls_task,
        dataset_name = 'nextsentence'
    )
    
    return {
        "train":{
            "mlm":tdata_mlm['train'],
            "qa":tdata_qa['train'],
            "sts":tdata_sts['train'],
            "cls":tdata_cls['train']
        },
        "val":{
            "mlm":tdata_mlm['val'],
            "qa":tdata_qa['val'],
            "sts":tdata_sts['val'],
            "cls":tdata_cls['val']
        }
    }
    
    
