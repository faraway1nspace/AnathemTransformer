from dataclasses import dataclass
from typing import Tuple, Union, Dict, List, Any,Optional

@dataclass
class TaskDataPerEpoch:
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    epoch:int
    index_stream:float
    taskname:str


class MLMDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for MLM task post-processing from huggingface stream."""
    train:List[str]
    val:List[str]
    epoch:int
    index_stream:float
    log_source:Dict[str]
    taskname:str = 'mlm'


class NextSentenceDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for NextSentence task post-processing from huggingface stream."""
    train:List[Dict[str,str]]
    val:List[Dict[str,str]]
    epoch:int
    index_stream:float
    log_source:Dict[str]    
    taskname:str = 'nextsentence'


class QADataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    epoch:int
    index_stream:float
    taskname:str = 'qa'


class STSDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    val:List[str]
    epoch:int
    index_stream:float
    taskname:str = 'sts'


class CLSDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[str]
    val:List[str]
    epoch:int
    index_stream:float
    taskname:str = 'cls'
    
    
