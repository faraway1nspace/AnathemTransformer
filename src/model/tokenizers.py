
import torch
from torch import nn, Tensor, device
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer
from transformers.utils import PaddingStrategy

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable

class CustomTokenizer:
    def __init__(
        self,
        model_string='google/bert_uncased_L-12_H-512_A-8',
        n_cls_prepend = 4,
        n_pad_to_multiple_of=4,
        downscale_multiple=2
    ):
        # initialize the tokenizer from the base model
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_string)
        # how many cls tokens to prepend to the fullsize data
        self.n_cls_prepend = n_cls_prepend
        self.n_pad_to_multiple_of = n_pad_to_multiple_of
        for k in dir(self.base_tokenizer):
            if not ((k[0]=='_') or (k in ['tokenize','encode','build_inputs_with_special_tokens','batch_encode_plus','encode_plus','pad'])):
                setattr(self,k,getattr(self.base_tokenizer, k))
        self.downscale_multiple = downscale_multiple
        # downscale attention
        self.maxpool_attn = nn.MaxPool1d(
            (self.downscale_multiple), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=True
        )

        # ensure excess_token_ids are included for .pad operations
        if 'excess_cls_ids' not in self.base_tokenizer.model_input_names:
            self.base_tokenizer.model_input_names += ['excess_cls_ids']

        for special_token_nm in [
            '_bos_token', '_cls_token', '_eos_token', '_mask_token', '_pad_token', '_pad_token_type_id', '_sep_token', '_unk_token'
        ]:
            setattr(self,special_token_nm, getattr(self.base_tokenizer, special_token_nm))

    def __call__(self, text, pad_to_multiple_of=None, add_special_tokens = True, return_tensors=None, *args, **kwargs):
        if pad_to_multiple_of is None:
            pad_to_multiple_of = self.n_pad_to_multiple_of
        tokens = self.base_tokenizer(
            text,
            pad_to_multiple_of=(pad_to_multiple_of if not add_special_tokens else 1),
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors if (not add_special_tokens) else None,
            *args,
            **kwargs
        )
        if add_special_tokens:
            tokens = self._batch_prepend_extra_cls_tokens_because_of_maxpooling(tokens, return_tensors)

        # downscale the attention, add to tokens
        tokens = self.downscale_attention(
            tokens, downscale_multiple=[self.downscale_multiple, self.downscale_multiple],name='attention_mask'
        )
        # dowscale the excess_cls_tokens, add to tokens
        tokens = self.downscale_attention(
            tokens, downscale_multiple=[self.downscale_multiple, self.downscale_multiple],name='excess_cls_ids'
        )
        return tokens

    def __len__(self):
        return len(self.base_tokenizer)

    def _num_pad_tokens(self, token_list):
        """Calculates how many PAD tokens to append to sequence to make a multiple of X"""
        return (self.n_pad_to_multiple_of - ((len(token_list)+(self.n_cls_prepend-1)) % self.n_pad_to_multiple_of)) % self.n_pad_to_multiple_of

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self.base_tokenizer._convert_id_to_token(index)

    def _prepend_extra_cls_tokens_because_of_maxpooling(self, tokens,return_tensors=None):
        n_cls_prepend = self.n_cls_prepend
        # prepend (n-1) CLS tokens to the front of the token_ids (because of maxpooling)
        # also pad so that the total length is a multiple of n_cls_prepend
        #num_pad_tokens = (self.n_pad_to_multiple_of - ((len_tokens+(n_cls_prepend-1)) % self.n_pad_to_multiple_of)) % self.n_pad_to_multiple_of
        tokens['input_ids'] = [self.cls_token_id]*(n_cls_prepend-1)+tokens['input_ids'] + [self.pad_token_id]*self._num_pad_tokens(tokens['input_ids'])
        tokens['excess_cls_ids'] = [0]*(n_cls_prepend)+tokens['attention_mask'][1:] +[0]*self._num_pad_tokens(tokens['attention_mask'])
        tokens['attention_mask'] = [1]*(n_cls_prepend-1)+tokens['attention_mask'] +[0]*self._num_pad_tokens(tokens['attention_mask'])
        if 'token_type_ids' in tokens.keys():
            tokens['token_type_ids'] = [
                tokens['token_type_ids'][0]
            ]*(n_cls_prepend-1) + tokens['token_type_ids'] + [tokens['token_type_ids'][-1]]*self._num_pad_tokens(tokens['token_type_ids'])
        if return_tensors == 'pt':
            for k,v in tokens.items():
                tokens[k] = torch.LongTensor(v)
        return tokens

    def _batch_prepend_extra_cls_tokens_because_of_maxpooling(self, tokens,return_tensors=None):
        n_cls_prepend = self.n_cls_prepend
        # prepend (n-1) CLS tokens to the front of the token_ids (because of maxpooling)
        # also pad so that the total length is a multiple of n_cls_prepend
        #num_pad_tokens = (self.n_pad_to_multiple_of - ((len_tokens+(n_cls_prepend-1)) % self.n_pad_to_multiple_of)) % self.n_pad_to_multiple_of
        tokens['input_ids'] = [
            [self.cls_token_id]*(n_cls_prepend-1)+input_id + [self.pad_token_id]*self._num_pad_tokens(input_id)
            for input_id
            in tokens['input_ids']
        ]
        tokens['excess_cls_ids'] = [
            [0]*(n_cls_prepend)+attnmask[1:] +[0]*self._num_pad_tokens(attnmask)
            for attnmask
            in tokens['attention_mask']
        ]
        tokens['attention_mask'] = [
            [1]*(n_cls_prepend-1)+attnmask +[0]*self._num_pad_tokens(attnmask)
            for attnmask
            in tokens['attention_mask']
        ]
        if 'token_type_ids' in tokens.keys():
            tokens['token_type_ids'] = [
                # we use the token_type_ids
                [toktypeid[0]]*(n_cls_prepend-1)+toktypeid +[toktypeid[-1]]*self._num_pad_tokens(toktypeid)
                for toktypeid
                in tokens['token_type_ids']
            ]
        if return_tensors == 'pt':
            for k,v in tokens.items():
                tokens[k] = torch.LongTensor(v)
        return tokens

    def encode(self, text, pad_to_multiple_of=4, add_special_tokens = True, *args, **kwargs):
        encoded = self.base_tokenizer.encode(text, pad_to_multiple_of=False, add_special_tokens=add_special_tokens, *args, **kwargs)
        if add_special_tokens:
            encoded = [self.cls_token_id]*(pad_to_multiple_of-1) + encoded
        if bool(pad_to_multiple_of):
            num_pad_tokens = (pad_to_multiple_of - (len(encoded) % pad_to_multiple_of)) % pad_to_multiple_of
            encoded += [self.pad_token_id] * num_pad_tokens
        return encoded

    def encode_plus(self, text, add_special_tokens=True, return_tensors=None, *args, **kwargs):
        tokens = self.base_tokenizer.encode_plus(text, add_special_tokens=add_special_tokens, return_tensors=return_tensors, *args, **kwargs)
        if add_special_tokens:
            tokens = self._prepend_extra_cls_tokens_because_of_maxpooling(tokens, return_tensors)
        return tokens

    def tokenize(self, text, add_special_tokens=True, *args, **kwargs):
        toks = self.base_tokenizer.tokenize(text, add_special_tokens=add_special_tokens, *args, **kwargs)
        if add_special_tokens:
            toks = [self.cls_token] * (self.n_cls_prepend-1) + toks
        return toks

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        out = self.base_tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        return [self.cls_token_id]*3 + out

    def batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        batched_encoded = self.base_tokenizer.batch_encode_plus( batch_text_or_text_pairs, *args, **kwargs)
        batched_encoded.update({'foo':'bar'})
        return batched_encoded

    def downscale_attention(self, tokens, downscale_multiple=None, name = 'attention_mask'):
        """
        Reduces the sequence-dimenion by self.downscale_multiple using nn.maxpool
        Adds the downscale attention to the tokens dictionary
        """
        if downscale_multiple is None:
            downscale_multiple = [self.downscale_multiple, self.downscale_multiple]

        # fullsize attention
        attn = tokens[name]
        if not isinstance(attn, torch.Tensor):
            attn = torch.Tensor(attn)

        for i, mult in enumerate(downscale_multiple):
            name_of_downsized_attn = '%s_l%d' % (name, i+2)
            with torch.no_grad():
                attn = self.maxpool_attn(attn.float())
            tokens[name_of_downsized_attn] = attn
        return tokens

    def pad(
        self,
        encoded_inputs,
        pad_to_multiple_of=4,
        return_tensors=None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        *args,
        **kwargs
    ):
        """Pad a list of tokenized-inputs to the same batch-length, with special processing of Anathem-specific inputs"""

        # which are conventional inputs and which are anathem specific
        conventional_input_nm = [k for k in encoded_inputs[0].keys() if k in ['input_ids', 'token_type_ids','attention_mask']]
        unconventional_input_nm = [k for k in encoded_inputs[0].keys() if k not in conventional_input_nm]

        # pad the vanilla inputs
        conventional_encoded_inputs = self.base_tokenizer.pad([
                {k:v for k,v in encoded_input.items() if k in conventional_input_nm}
                for encoded_input in encoded_inputs
            ], pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors, padding=padding, max_length=max_length, *args, **kwargs
        )

        # deal with the remaining inputs
        padding_strategy, _, max_length, _ = self.base_tokenizer._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=False
        )

        #required_input = encoded_inputs[][self.model_input_names[0]]
        # this is stupid, I need to pad each input in batch individually
        special_anathem_inputs = [
                {k:v for k,v in encoded_input.items() if k in unconventional_input_nm}
                for encoded_input in encoded_inputs
        ]
        special_anathem_encoded_inputs = self.pad_special_anathem_inputs(
            special_anathem_inputs=special_anathem_inputs,
            encoded_inputs=conventional_encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,#: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors
        )
        # let's see if I can just insert into the conventional_encode_inputs
        conventional_encoded_inputs.update(special_anathem_encoded_inputs) # apparently I can just append..

        # downscale the attention and add to inputs
        conventional_encoded_inputs = self.downscale_attention(
            conventional_encoded_inputs,
            downscale_multiple=[self.downscale_multiple, self.downscale_multiple],
            name='attention_mask'
        )
        # dowscale the excess_cls_tokens, add to tokens
        conventional_encoded_inputs = self.downscale_attention(
            conventional_encoded_inputs,
            downscale_multiple=[self.downscale_multiple, self.downscale_multiple],
            name='excess_cls_ids'
        )
        return conventional_encoded_inputs

    def pad_special_anathem_inputs(
        self,
        special_anathem_inputs,
        encoded_inputs,
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors=None,
    ):
        required_input = encoded_inputs[self.model_input_names[0]]
        batch_size,max_length = required_input.shape
        #print(batch_size,max_length)
        assert batch_size == len(special_anathem_inputs)
        assert isinstance(special_anathem_inputs, list)
        padding_strategy = PaddingStrategy.MAX_LENGTH
        special_anathem_batch_outputs = {}
        for i in range(batch_size):
            inputs = special_anathem_inputs[i] #{k: v[i] for k, v in special_anathem_inputs.items()}
            assert isinstance(inputs, dict)
            outputs = self._pad_special_anathem_input(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of
            )
            for key, value in outputs.items():
                if key not in special_anathem_batch_outputs:
                    special_anathem_batch_outputs[key] = []
                special_anathem_batch_outputs[key].append(value)

        return BatchEncoding(special_anathem_batch_outputs, tensor_type=return_tensors) # returning because of failure

    def _pad_special_anathem_input(
        self,
        special_anathem_input,
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None
    ) -> dict:
        """
        Pad encoded Anathem-specific inputs (on left/right and up to predefined length or max length in the batch)
        """
        assert isinstance(special_anathem_input, dict)
        len_required_input = len(special_anathem_input[list(special_anathem_input.keys())[0]])
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len_required_input != max_length

        # Initialize attention mask if not present
        if needs_to_be_padded:
            special_anathem_outputs = dict.fromkeys(special_anathem_input.keys())
            difference = max_length - len_required_input
            if self.padding_side == "right":
                for k in special_anathem_input.keys():
                    special_anathem_outputs[k] = special_anathem_input[k] + [0] * difference
            elif self.padding_side == "left":
                for k in special_anathem_input.keys():
                    special_anathem_outputs[k] = [0] * difference + special_anathem_input[k]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

            return special_anathem_outputs
        return special_anathem_input