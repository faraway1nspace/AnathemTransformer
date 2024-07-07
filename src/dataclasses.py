from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, List, Any,Optional

@dataclass
class TaskDataPerEpoch:
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    epoch:int
    index_stream:float
    log_source:Optional[dict]
    taskname:str


@dataclass
class MLMDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for MLM task post-processing from huggingface stream."""
    train:List[str]
    val:List[str]
    epoch:int
    index_stream:float
    log_source:Dict[str, Dict[str,float]]
    taskname:str = 'mlm'


@dataclass
class NextSentenceDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for NextSentence task post-processing from huggingface stream."""
    train:List[Dict[str,str]]
    val:List[Dict[str,str]]
    epoch:int
    index_stream:float
    log_source:Dict[str, Dict[str,float]]
    taskname:str = 'nextsentence'


@dataclass
class QADataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    epoch:int
    index_stream:float
    log_source:Optional[dict]
    taskname:str = 'qa'


@dataclass
class STSDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[Dict[str,Union[str,List[str]]]]
    val:List[Dict[str,Union[str,List[str]]]]
    val:List[str]
    epoch:int
    index_stream:float
    log_source:Optional[dict]
    taskname:str = 'sts'


@dataclass
class CLSDataPerEpoch(TaskDataPerEpoch):
    """Train & val data for QA task post-processing from huggingface stream."""    
    train:List[str]
    val:List[str]
    epoch:int
    index_stream:float
    log_source:Optional[dict]
    taskname:str = 'cls'
    
    
