import torch
from torch import Tensor, device
from transformers import AutoModel, AutoTokenizer
from typing import Dict,Any,List

from src.model.model_utils import batch_to_device
from src.configs.constants import MAX_SEQ_LENGTH, SEED

class TeacherEmbedder:
    """Wrapper for an sbert Teacher embedding model for istillation."""
    def __init__(
        self,
        pretrained_name = 'mixedbread-ai/mxbai-embed-large-v1', # 'intfloat/e5-large-v2'
        device:device=None,
        query_prefix:str = "Represent this sentence for searching relevant passages: ",
        passage_prefix:str="",
        pooling:str='cls'
    ): 
        self.pretrained_name = pretrained_name
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.teacher_embedder = AutoModel.from_pretrained(pretrained_name)
        self._target_device = torch.device('cpu') if device is None else device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling_method = pooling
        assert pooling in ['cls','mean']

    @staticmethod
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def cls_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return last_hidden_states[:, 0]

    def pooling(self, last_hidden_states: Tensor, attention_mask: Tensor)->Tensor:
        """Pooling methods to convert sequence embedding to paragraph/sent embedding, by CLS token or mean pooling."""
        if self.pooling_method == 'cls':
            return self.cls_pool(last_hidden_states, attention_mask)
        return self.average_pool(last_hidden_states, attention_mask)

    def forward(self, input_text:List[str], prepend:str = "", prepend_type:str|None=None):
        if prepend_type is not None and prepend_type=='query':
            prepend = self.query_prefix
        elif prepend_type is not None and prepend_type in ['pos','neg','passage']:
            prepend = self.passage_prefix
        input_text = [prepend + s for s in input_text]
        with torch.no_grad():
            batch_dict = self.teacher_tokenizer(input_text, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_to_device(batch_dict, self._target_device)
            outputs = self.teacher_embedder(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings

    def to(self,device:device):
        self._target_device = device
        self.teacher_embedder.to(device)

    def __call__(self, input_text, prepend:str = "", prepend_type:str|None=None):
        return self.forward(input_text, prepend,prepend_type)

    def eval(self):
        self.teacher_embedder.eval()



#teacher_emb = TeacherEmbedder(pretrained_name = 'mixedbread-ai/mxbai-embed-large-v1', pooling='cls')
#teacher_emb(["This is a test sentence to embed","The tenant is allowed 30 days to find a new appartment"],prepend_type='query')