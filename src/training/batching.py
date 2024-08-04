import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

from src.configs.constants import MAX_SEQ_LENGTH,SEED
from src.model.tokenizers import CustomTokenizer

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable


class DataCollatorForWholeWordMaskAnamod:
    """Modified version of DataCollatorForWholeWordMask adapted for Anathem's Tokenizer."""
    def __init__(
        self,
        tokenizer:CustomTokenizer,
        mlm:bool=True,
        mlm_probability:float=0.12
    ):
        self.tokenizer = tokenizer
        self.data_collator_mlm = DataCollatorForWholeWordMask(
            tokenizer=self.tokenizer, mlm=mlm, mlm_probability=mlm_probability
        )

    @staticmethod
    def _dict_of_inputs_to_list_of_dict_of_examples(tokenized_inputs:Dict[str,Any])->List[Union[List[int], Any, Dict[str, Any]]]:
        """Converts a dictionary of inputs into a list of dict inputs for the DataCollator to function."""
        tokenized_input_list = [
            {'input_ids': input_ids, 'attention_mask': attention_mask}
            for input_ids, attention_mask in zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
        ]
        return tokenized_input_list

    def __call__(self, batch:List[str])->Dict[str, Union[Tensor,str]]:
        """Special wrapper for DataCollatorForWholeWordMask for the anamod.tokenizer"""
        tokenized_input = self.data_collator_mlm.tokenizer(
                batch,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH-4,
                truncation=True,
                padding='longest',  # Dynamically pad to the longest sequence in the batch
                return_tensors='pt'
            )

        # Use the data collator to create masked tokens and targets
        mlm_batch_collated = self.data_collator_mlm(self._dict_of_inputs_to_list_of_dict_of_examples(tokenized_input))

        # update the (masked) input IDs and newly created MLM labels
        assert mlm_batch_collated['input_ids'].shape[0] == tokenized_input['attention_mask'].shape[0]
        assert mlm_batch_collated['input_ids'].shape[1] == tokenized_input['attention_mask'].shape[1]
        tokenized_input['input_ids'] = mlm_batch_collated['input_ids']
        tokenized_input['labels'] = mlm_batch_collated['labels']
        tokenized_input['text'] = batch
        return tokenized_input


class LengthBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset:Dataset,
        batch_size:int,
        drop_last:bool=True,
        get_item_len:Callable= (lambda x: len(x)),
        seed:int = SEED,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.get_item_len_fn = get_item_len
        self.seed = seed
        self.batches = self.create_batches()

    def create_batches(self):
        # Get lengths of all texts
        lengths = [self.get_item_len_fn(item) for item in self.dataset]

        # Get sorted indices based on length
        sorted_indices = np.argsort(lengths)

        # Create batches of indices
        batches = [sorted_indices[i:i + self.batch_size] for i in range(0, len(sorted_indices), self.batch_size)]

        # Shuffle batches
        np.random.RandomState(self.seed).shuffle(batches)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.batches) - 1
        return len(self.batches)