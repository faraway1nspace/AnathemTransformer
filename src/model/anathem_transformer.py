import copy
import math
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn, Tensor, device
import torch.nn.functional as F
from torch.cuda import is_available
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.configuration_bert import BertConfig
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PaddingStrategy
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable

from src.model.tokenizers import CustomTokenizer

class BertSelfAttnDimensionReduction(nn.Module):
    """Bert Attention Layer that uses a dimension-reduced version of the query, so to reduce the dimension of the outputs"""
    def __init__(
        self,
        config,
        hidden_size_input=768,
        hidden_size_query = None,
        position_embedding_type=None,
        dim_reduction = 2
    ):
        """Special type of Bert Self attention that reduces the dimension of the inputs by half"""
        super().__init__()
        if (config.hidden_size // dim_reduction) % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.dim_reduction = dim_reduction
        self.hidden_size_input = hidden_size_input
        self.hidden_size_reduced = hidden_size_input // dim_reduction
        if hidden_size_query is None:
            hidden_size_query = hidden_size_input
        self.hidden_size_query = hidden_size_query
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(self.hidden_size_reduced / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size_query, self.all_head_size)
        self.key = nn.Linear(self.hidden_size_input, self.all_head_size)
        self.value = nn.Linear(self.hidden_size_input, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.

        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if encoder_attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            #print(attention_scores.shape)
            #print(attention_scores.shape)
            attention_scores = attention_scores + encoder_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class InterpolateCombo(nn.Module):
    """there could also be an attentive way to do this"""
    def __init__(self, scale_factor=2, dropout=0.05, alpha=0.667):
        """Arguments:
        :param scaler_factor: float, multiple of up-scaling
        :param dropout: float, dropout proportion
        :param alpha: float, mixture weight between nearest-neighbor vs linear-interpolation
        """
        super(InterpolateCombo, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)
        self.a = alpha

    def forward(self, x):
        x_trans = x.transpose(-2,-1)
        z = self.a*self.interp(x_trans, mode='nearest',scale_factor=self.scale_factor) + (1-self.a)*self.interp(x_trans, mode='linear',scale_factor=self.scale_factor)
        z = self.dropout(z)
        return z.transpose(-2,-1)


class BertCrossAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
        hidden_size_query,
        hidden_size_keyvalue=None,
        position_embedding_type=None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_query = hidden_size_query
        if hidden_size_keyvalue is None:
            hidden_size_keyvalue = hidden_size
        self.hidden_size_keyvalue = hidden_size_keyvalue
        if self.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(self.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size_query, self.all_head_size)
        self.key = nn.Linear(self.hidden_size_keyvalue, self.all_head_size)
        self.value = nn.Linear(self.hidden_size_keyvalue, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        query_hidden_states: Optional[torch.FloatTensor] = None,
        query_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(query_hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertReduceAddIntegrativeLayer(nn.Module):
    """Bert Layer that does dimenion reduction along embedding-dimenion and integrations a skip connection"""
    def __init__(
            self,
            config,
            hidden_size,
            hidden_size_input=None,
            hidden_size_query=None,
            intermediate_size=None,
            dim_reduction=2,
            do_concat_hidden_and_query = True
        ):
        super().__init__()
        #self.chunk_size_feed_forward = config.chunk_size_feed_forward
        #self.seq_len_dim = 1
        self.cat = torch.cat
        self.do_concat_hidden_and_query = do_concat_hidden_and_query
        assert bool(do_concat_hidden_and_query), 'not implemented: concatenation of query and hidden-states must happen'
        self.hidden_size = hidden_size
        if dim_reduction is None:
            dim_reduction = 2
        self.dim_reduction = dim_reduction
        if intermediate_size is None:
            intermediate_size = int(4*hidden_size)
        self.intermediate_size = intermediate_size
        if hidden_size_input is None:
            hidden_size_input = hidden_size
        self.hidden_size_input = hidden_size_input
        if hidden_size_query is None:
            hidden_size_query = hidden_size_input
        self.hidden_size_query = hidden_size_query + do_concat_hidden_and_query*hidden_size
        self.hidden_size_concat = int(hidden_size + hidden_size_input)

        # cross attention between (low-res) query and hidden layers below
        self.attention = BertSelfAttnDimensionReduction(
            config,
            hidden_size_input=self.hidden_size_input,
            hidden_size_query = self.hidden_size_query,
            position_embedding_type="absolute",
            dim_reduction = self.dim_reduction
        )
        self.is_decoder = config.is_decoder
        #inputs = x_l1, x_l1_reduced, x_l2_prev
        #- x2 = BertCrossAttention(k,v=x_l1, q= cat(x_l1_reduced, x_l2_prev) ) -notice three inputs
        #- x3 = lnorm(drop(f(x2)) + x_l2_prev)
        #- x4_ex = activation( f(cat(x3, x_l1_reduced))  )
        #- x5 = lnorm(drop(f(x4_ex)) + x3)

        # corresponds to BertAttention SelfOutput
        self.output_attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.lnorm_attn = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout_attn = nn.Dropout(config.hidden_dropout_prob)

        # corresponds to BertIntermediate
        self.intermediate = nn.Linear(self.hidden_size_concat, self.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # corresponds to BertOutput
        self.output_intm = nn.Linear(self.intermediate_size, self.hidden_size)
        self.lnorm_intm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout_intm = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor, # higher-resolution inputs for key and values (long sequence dimension)
        hidden_states: torch.Tensor, # previous hidden-states for skip connection (short squence-dim, low-res)
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        query_hidden_states: torch.FloatTensor = None, # hidden-states for query (short squence-dim, low-res)
        query_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        if self.do_concat_hidden_and_query:
            query_hidden_states_plus = torch.cat((query_hidden_states, hidden_states),axis=2)
        # cross attn between (low-res) query vector and (high-res) key-values
        cross_attn_outputs = self.attention(
            query_hidden_states_plus, # query (short seq-dim, high-res)
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states = inputs, # for key/value (longer sequence dimension, high-res)
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        cross_hidden_states = cross_attn_outputs[0]

        # first Add+Norm skip connection (BertSelfOutput)
        cross_hidden_states = self.dropout_attn(self.output_attn(cross_hidden_states))
        hidden_states = self.lnorm_attn(cross_hidden_states + hidden_states)

        # intermediate expension
        intermediate_states = self.intermediate_act_fn(self.intermediate(
            self.cat((hidden_states, query_hidden_states),axis=2)
        ))
        assert intermediate_states.shape[0]==hidden_states.shape[0]
        assert intermediate_states.shape[1]==hidden_states.shape[1]

        # BertOutput
        intermediate_states = self.dropout_intm(self.output_intm(intermediate_states))
        out_states = self.lnorm_intm(intermediate_states + hidden_states)

        #inputs = x_l1, x_l1_reduced, x_l2_prev
        #- x2 = BertCrossAttention(k,v=x_l1, q= cat(x_l1_reduced, x_l2_prev) ) -notice three inputs
        #- x3 = lnorm(drop(f(x2)) + x_l2_prev)
        #- x4_ex = activation( f(cat(x3, x_l1_reduced))  )
        #- x5 = lnorm(drop(f(x4_ex)) + x3)
        return out_states

try:
    from transformers.modeling_utils import get_extended_attention_mask
except:
    print('WARNING: could not load `get_extended_attention_mask` from transformers.modeling_utils')
    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: device) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class BertIntegrativeLayer(nn.Module):
    """Vanilla Bert Layer, but integrates other hiddens states from a parallel transformers stack typically low-res
    BertIntegrativeLayer:
    - x2 = BertCrossAttention(k,v=x_l2, q=x_l3_up)
    - x3 = lnorm(drop(f(x2)) + x_l2)
    - x4_ex = activation( f(cat(x3, x_l3_up))  )
    - x5 = lnorm(drop(f(x4_ex)) + x3)
    """
    def __init__(
            self,
            config,
            hidden_size, # dimensions of the (high-res) hiddens states; same dimension as output
            hidden_size_keyvalues, # dimensions of (low-res) states used as key/values; 1/2 sequence-length and dim
            hidden_size_query_to_concat=None, # upsampled (low-res,1/2 seq-dim) to concat to hidden_states (high-res)
            intermediate_size=None
        ):
        super().__init__()
        #self.chunk_size_feed_forward = config.chunk_size_feed_forward
        #self.seq_len_dim = 1
        self.cat = torch.cat
        self.hidden_size = hidden_size
        self.hidden_size_keyvalues = hidden_size_keyvalues
        if hidden_size_query_to_concat is None:
            hidden_size_query_to_concat = hidden_size_keyvalues
        self.hidden_size_query_to_concat = hidden_size_query_to_concat
        self.hidden_size_query = int(hidden_size + hidden_size_query_to_concat)
        self.hidden_size_concat = int(hidden_size + hidden_size_query_to_concat)
        if intermediate_size is None:
            intermediate_size = int(4*hidden_size)
        self.intermediate_size = intermediate_size

        # cross attention between (low-res) query and hidden layers below
        self.attention = BertCrossAttention(
            config,
            hidden_size= self.hidden_size, # high dim output
            hidden_size_query = self.hidden_size_query, # high dim query
            hidden_size_keyvalue = self.hidden_size_keyvalues, # low-dim keyvalues
            position_embedding_type="absolute"
        )
        self.is_decoder = config.is_decoder
        #self.intermediate = BertIntermediate(config)
        #self.output = BertOutput(config)
        #- x2 = BertCrossAttention(k,v=x_l2, q=x_l3_up)
        #- x3 = lnorm(drop(f(x2)) + x_l2)
        #- x4_ex = activation( f(cat(x3, x_l3_up))  )
        #- x5 = lnorm(drop(f(x4_ex)) + x3)

        # corresponds to BertAttention SelfOutput
        self.output_attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.lnorm_attn = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout_attn = nn.Dropout(config.hidden_dropout_prob)

        # corresponds to BertIntermediate
        self.intermediate = nn.Linear(self.hidden_size_concat, self.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # corresponds to BertOutput
        self.output_intm = nn.Linear(self.intermediate_size, self.hidden_size)
        self.lnorm_intm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout_intm = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor, # high-res hidden states (same dimensions as output), used as query
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        keyvalue_hidden_states: torch.Tensor=None, # low-res hidden-states (1/2 seq-dim) used for key-value pairs
        query_to_concat_hidden_states: torch.Tensor=None, # to concatenate to query (I think this is upsampled from low-res 1/2 seq-dim)
        query_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # cross attn between hiddens states and (low-res) query vector
        cross_attn_outputs = self.attention(
            hidden_states = keyvalue_hidden_states,
            attention_mask = attention_mask,
            head_mask = head_mask,
            query_hidden_states = torch.cat((hidden_states, query_to_concat_hidden_states),axis=2),
            query_attention_mask = query_attention_mask
        )
        cross_hidden_states = cross_attn_outputs[0]
        assert cross_hidden_states.shape[1]==hidden_states.shape[1], f"{cross_hidden_states.shape[1]},{cross_hidden_states.shape[2]} vs {hidden_states.shape[1]},{hidden_states[2]}"
        assert cross_hidden_states.shape[2]==hidden_states.shape[2]


        # first Add+Norm skip connection (BertSelfOutput)
        cross_hidden_states = self.output_attn(cross_hidden_states)
        cross_hidden_states = self.dropout_attn(cross_hidden_states)
        hidden_states = self.lnorm_attn(cross_hidden_states + hidden_states)

        # intermediate expension
        intermediate_states = self.cat((hidden_states, query_to_concat_hidden_states),axis=2)
        intermediate_states = self.intermediate(intermediate_states)
        intermediate_states = self.intermediate_act_fn(intermediate_states)
        assert intermediate_states.shape[0]==hidden_states.shape[0]
        assert intermediate_states.shape[1]==hidden_states.shape[1]

        # BertOutput
        out_states = self.output_intm(intermediate_states)
        out_states = self.dropout_intm(out_states)
        out_states = self.lnorm_intm(out_states + hidden_states)

        #- x2 = BertCrossAttention(k,v=x_l2, q=x_l3_up)
        #- x3 = lnorm(drop(f(x2)) + x_l2)
        #- x4_ex = activation( f(cat(x3, x_l3_up))  )
        #- x5 = lnorm(drop(f(x4_ex)) + x3)
        return out_states


class CheapMLPIntegrativeLayer(nn.Module):
    """Cheap (non-attentive) Integrator layer that merges a low-res layers with higher-res layer."""
    def __init__(
            self,
            config,
            hidden_size, # dimensions of the (high-res) hiddens states; same dimension as output
            hidden_size_keyvalues=None, # dimensions of (low-res) states used as key/values; 1/2 sequence-length and dim
            hidden_size_query_to_concat=None, # dimensions of (low-res) to concat to hidden_states; 1/2 sequence-length and dim
            intermediate_size=None
        ):
        super().__init__()
        #self.chunk_size_feed_forward = config.chunk_size_feed_forward
        #self.seq_len_dim = 1
        self.cat = torch.cat
        self.hidden_size = hidden_size
        if hidden_size_keyvalues is None:
            hidden_size_keyvalues = hidden_size
        self.hidden_size_keyvalues = hidden_size_keyvalues
        if hidden_size_query_to_concat is None:
            hidden_size_query_to_concat = hidden_size_keyvalues
        self.hidden_size_query_to_concat = hidden_size_query_to_concat
        self.hidden_size_query = int(hidden_size + hidden_size_query_to_concat)
        if intermediate_size is None:
            intermediate_size = int(2*hidden_size)
        self.intermediate_size = intermediate_size

        # expand hidden-size to a multiple
        self.dense_expander = nn.Linear(
            self.hidden_size_query,
            self.intermediate_size
        ) # deflate back to same size as hidden-state
        self.dense_deflator = nn.Linear(
            self.intermediate_size,
            self.hidden_size
        )

        # intermediate activation function
        self.intermediate_act_fn = nn.RReLU(0.0625, 0.125)

        # corresponds to BertOutput
        self.lnorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor, # high-res hidden states (same dimensions as output), used as query
        attention_mask = None, # ignored
        head_mask = None, # ignored
        keyvalue_hidden_states =None, # ignored
        query_to_concat_hidden_states: torch.Tensor=None, # to concatenate to hidden_states
        query_attention_mask = None, # ignored
        past_key_value = None, # ignored
        output_attentions = False, # ignored
    ) -> torch.Tensor:

        # concat (lowres) to hidden-states
        inputs = self.cat((hidden_states, query_to_concat_hidden_states),axis=2)
        # expand x2 dimension
        intermediate_states = self.dense_expander(inputs)
        # activation (leaky relue)
        intermediate_states = self.intermediate_act_fn(intermediate_states)
        # like BertOutput
        out_states = self.dense_deflator(intermediate_states)
        # dropout
        out_states = self.dropout(out_states)
        # combine with hidden-state inputs
        out_states = self.lnorm(out_states + hidden_states)

        return out_states



def make_config_anathem(
    modelstring:str = "distilroberta-base",
    num_transformer_stacks:int = 3,
    scale_ratio2:float = 0.5,
    scale_ratio3:float = 0.25,
    multiplier_intermediate2:float = 4.0,
    multiplier_intermediate3:float = 4.0,
    num_layers_l2:int = 1, # mid-res encoder
    num_layers_l3:int = 3, # low-res encoder
    dropout_scaling:float = 0.05,
    do_cheap_integrator:List[int] = [1],
    sequence_classification_intermediate_dim = None, # default is the same as the basemodel hidden-dim
    sequence_classification_out_dim = None, # default is x2 same as the basemodel hidden-dim
    do_mlm:bool =False,
    do_cls:bool = False,
    do_pair_cls:bool = True,
    n_labels:int = 24
)->Union[PretrainedConfig, BertConfig]:
    #if True:
    #modelstring = "distilroberta-base"
    #scale_ratio2 = 0.5
    #scale_ratio3 = 0.25
    #scale_intermediate2 = 4
    #scale_intermediate3 = 4
    base_config = AutoConfig.from_pretrained(modelstring)
    config_l2 = copy.deepcopy(base_config)
    config_l3 = copy.deepcopy(base_config)
    setattr(base_config, 'model_string', modelstring)
    setattr(base_config,'num_transformer_stacks', num_transformer_stacks)
    setattr(base_config,'num_layers_l2', num_layers_l2)
    setattr(base_config,'num_layers_l3', num_layers_l3)
    setattr(base_config,'scale_ratio2', scale_ratio2)
    setattr(base_config,'scale_ratio3', scale_ratio3)
    setattr(base_config,'scale_factor2', int(1/base_config.scale_ratio2))
    setattr(base_config,'scale_factor3', int(1/base_config.scale_ratio3*base_config.scale_ratio2))
    setattr(base_config,"hidden_size_l2", int(base_config.hidden_size * scale_ratio2))
    setattr(base_config,"hidden_size_l3", int(base_config.hidden_size * scale_ratio3))
    setattr(base_config,"intermediate_size_l1", int(base_config.hidden_size_l2*multiplier_intermediate2))
    setattr(base_config,"intermediate_size_l2", int(base_config.hidden_size_l3*multiplier_intermediate3))
    setattr(base_config,"query_size1", base_config.hidden_size_l2 + base_config.hidden_size_l3)
    setattr(base_config,"query_size2", base_config.hidden_size_l3)
    setattr(base_config,"dropout_scaling", dropout_scaling)
    setattr(base_config,"use_cheap_integrator_for_stacks", do_cheap_integrator)
    setattr(base_config, "do_mlm", do_mlm)
    setattr(base_config, "do_cls", do_cls)
    setattr(base_config, "do_pair_cls", do_pair_cls)
    setattr(base_config, "n_labels", n_labels)

    # hidden dimension
    setattr(
        base_config,
        "sequence_classification_intermediate_dim",
        sequence_classification_intermediate_dim  if sequence_classification_intermediate_dim is not None else [
            int(base_config.hidden_size*s)
            for s in [1, scale_ratio2, scale_ratio3]
        ]
    )
    # final dimension outputed for sequence classification
    setattr(
        base_config,
        "sequence_classification_out_dim",
        sequence_classification_out_dim  if sequence_classification_out_dim is not None else base_config.hidden_size*2
    )

    # make the configuration for the l2 mid-res encoder
    config_l2.hidden_size = base_config.hidden_size_l2
    config_l2.num_hidden_layers = num_layers_l2
    setattr(base_config, 'config_l2', config_l2)

    # make the configuration for the l3 encoder
    config_l3.hidden_size = base_config.hidden_size_l3
    config_l3.num_hidden_layers = num_layers_l3
    setattr(base_config, 'config_l3', config_l3)
    return base_config


def initialize_baselayers(config, basemod = None, tokenizer=None, stack_id=0):
    """Initializes the embeddings and first stack of layers for the Anathem transformers"""
    # initialize the basemodel
    if basemod is None:
        basemod = AutoModel.from_pretrained(config.model_string)
    if tokenizer is None:
        # download pretrained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_string)

    device = basemod.device
    setattr(config, 'device', device)

    # get basemodel's embeddings
    layer_embedding = copy.deepcopy(basemod._modules['embeddings'])

    # get basemodel's first transformer block
    layer_basetransformer = copy.deepcopy(basemod._modules['encoder']._modules['layer']._modules['0'])

    # initialize the maxpooling downsamplers
    maxpool = nn.Sequential(
        nn.Dropout(config.dropout_scaling),
        nn.MaxPool2d((2,1), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=True)
    )
    # pooling the attention has no dropout
    maxpool_attn = nn.MaxPool1d((2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=True)

    # initialize downsampling attention layers
    bert_reducer_l2 = BertSelfAttnDimensionReduction(
        config=config,
        hidden_size_input=config.hidden_size,
        position_embedding_type=config.position_embedding_type,
        dim_reduction = config.scale_factor2
    )
    # 1/4 hidden size
    bert_reducer_l3 = BertSelfAttnDimensionReduction(
        config=config,
        hidden_size_input=config.hidden_size_l2,
        position_embedding_type=config.position_embedding_type,
        dim_reduction = config.scale_factor3
    )

    # initialize the mid-resolution BertEncoder
    bert_encoder_midres = BertEncoder(config.config_l2)
    # initialize the low-resolution BertEncoder
    bert_encoder_lowres = BertEncoder(config.config_l3)

    # initailize the upscalers
    upscaler_x2 = InterpolateCombo(scale_factor=config.scale_factor3, dropout=config.dropout_scaling)
    upscaler_x4 = InterpolateCombo(scale_factor=int(1/config.scale_ratio3), dropout=config.dropout_scaling)

    # initialize the BertIntegrative Layers: low res to mid res
    bert_integrater_l2 = BertIntegrativeLayer(
        config,
        hidden_size=config.hidden_size_l2,
        hidden_size_keyvalues = config.hidden_size_l3,
        hidden_size_query_to_concat=config.hidden_size_l3,
        intermediate_size=config.intermediate_size_l2
    )

    # from mid-res to high-res
    do_cheap_integrator = (stack_id in config.use_cheap_integrator_for_stacks)
    # from mid-res to high-res
    if not do_cheap_integrator:
        bert_integrater_l1 = BertIntegrativeLayer(
            config,
            hidden_size=config.hidden_size,
            hidden_size_keyvalues = config.hidden_size_l2,
            hidden_size_query_to_concat=config.hidden_size_l2,
            intermediate_size=config.intermediate_size_l1
        )
    else:
        bert_integrater_l1 = CheapMLPIntegrativeLayer(
            config,
            hidden_size=config.hidden_size,
            hidden_size_query_to_concat=config.hidden_size_l2,
            intermediate_size=config.hidden_size*2
        )

    return (
        tokenizer,
        basemod,
        layer_embedding,
        layer_basetransformer,
        maxpool,
        maxpool_attn,
        bert_reducer_l2,
        bert_reducer_l3,
        bert_encoder_midres,
        bert_encoder_lowres,
        upscaler_x2,
        upscaler_x4,
        bert_integrater_l2,
        bert_integrater_l1
    )


def initialize_midlayers(
    config,
    basemod=None,
    tokenizer=None,
    stack_id=1
):
    """Initializes all the intermediate layers for settup-up the Anathem transformer model."""
    # initialize the maxpooling downsamplers
    maxpool = nn.Sequential(
        nn.Dropout(config.dropout_scaling),
        nn.MaxPool2d((2,1), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=True)
    )
    # pooling the attention has no dropout
    maxpool_attn = nn.MaxPool1d((2), stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=True)

    # initialize bert attentive downsampling and skipconnection (1/2 embedding dim)
    bert_reduceintegrator_l2 = BertReduceAddIntegrativeLayer(
        config,
        config.hidden_size_l2, # size of mid-res
        hidden_size_input=config.hidden_size, # size full-resolution
        hidden_size_query=config.hidden_size, # size full-resolution
        intermediate_size=config.intermediate_size_l1, # BertIntermediate dimension (expansion *4 the hiddensize)
        dim_reduction=config.scale_factor2, # reduce embedding dimension by factor of 2
        do_concat_hidden_and_query = True
    )

    # 1/4 the size
    bert_reduceintegrator_l3 = BertReduceAddIntegrativeLayer(
        config,
        config.hidden_size_l3, # size of mid-res
        hidden_size_input=config.hidden_size_l2, # size full-resolution
        hidden_size_query=config.hidden_size_l2, # size full-resolution
        intermediate_size=config.intermediate_size_l2, # BertIntermediate dimension
        dim_reduction=config.scale_factor3, # reduce embedding dimension by factor of 2
        do_concat_hidden_and_query = True
    )

    # initialize the low-resolution BertEncoder
    bert_encoder_midres = BertEncoder(config.config_l2)
    bert_encoder_lowres = BertEncoder(config.config_l3)

    # initailize the upscalers
    upscaler_x2 = InterpolateCombo(scale_factor=config.scale_factor3, dropout=config.dropout_scaling)
    upscaler_x4 = InterpolateCombo(scale_factor=int(1/config.scale_ratio3), dropout=config.dropout_scaling)

    # initialize the BertIntegrative Layers: from low-res to mide-res
    bert_integrater_l2 = BertIntegrativeLayer(
        config,
        hidden_size=config.hidden_size_l2,
        hidden_size_keyvalues = config.hidden_size_l3,
        hidden_size_query_to_concat=config.hidden_size_l3,
        intermediate_size=config.intermediate_size_l2
    )

    do_cheap_integrator = (stack_id in config.use_cheap_integrator_for_stacks)
    if not do_cheap_integrator:
        # from mid-res to high-res
        bert_integrater_l1 = BertIntegrativeLayer(
            config,
            hidden_size=config.hidden_size,
            hidden_size_keyvalues = config.hidden_size_l2,
            hidden_size_query_to_concat=config.hidden_size_l2,
            intermediate_size=config.intermediate_size_l1
        )
    else:
        bert_integrater_l1 = CheapMLPIntegrativeLayer(
            config,
            hidden_size=config.hidden_size,
            hidden_size_query_to_concat=config.hidden_size_l2,
            intermediate_size=config.hidden_size*2
        )

    return (
        maxpool,
        maxpool_attn,
        bert_reduceintegrator_l2,
        bert_reduceintegrator_l3,
        bert_encoder_midres,
        bert_encoder_lowres,
        upscaler_x2,
        upscaler_x4,
        bert_integrater_l2,
        bert_integrater_l1
    )


def initialize_finaltransformerlayers(
    config,
    basemod=None,
    tokenizer=None,
    names_encoder_module = 'encoder',
    stack_id:int=3
):
    """Initializes the final BertLayer before output, but copying the final BertLayer from `Basemod`.
    Basemod is just the Embedding, Tokenizer and first Bertlayers from another transformer model available on Hugggingface."""
    # initialize the maxpooling downsamplers
    assert basemod is not None, "`initialize_finaltransformerlayers` requires the basemod to instantiate the final transformer block"

    # get the Encoder stacks
    assert names_encoder_module in basemod._modules.keys(), 'expected %s in basemod._modules' % names_encoder_module
    basemod_encoder_stack = get_to_bertlayer(basemod, target_layer_name = names_encoder_module)

    # get the name of the final transformer block (-1) in encoder
    names_of_final_transformer_block = list(basemod_encoder_stack._modules['layer']._modules.keys())[-1]

    # get the final transformer block (NN weights pretrained)
    bert_finaltransformer_block = basemod_encoder_stack._modules['layer']._modules[
        names_of_final_transformer_block
    ]

    return copy.deepcopy(bert_finaltransformer_block)

def get_to_bertlayer(
    basemod,
    target_layer_name:str = 'encoder',
    model_string:Optional[str]=None
):
    """Locates a particular layer name within a pretrained bert model."""
    if  target_layer_name in basemod._modules.keys():
        return basemod._modules[target_layer_name]
    elif target_layer_name in basemod._modules['bert']._modules.keys():
        return basemod._modules['bert']


class AnathemBaseModule(nn.Module):
    """First Sstack of layers with embeddings, that go full circle form high-res to low-res back to high res"""
    def __init__(
            self,
            config,
            basemod=None,
            tokenizer=None,
            past_key_values_length = None,
            device = None,
            stack_id=0
        ):
        super().__init__()
        self.config = config

        # initalize the layers
        (
            tokenizer, basemod,
            layer_embedding,
            layer_basetransformer,
            maxpool,
            maxpool_attn,
            bert_reducer_l2,
            bert_reducer_l3,
            bert_encoder_midres,
            bert_encoder_lowres,
            upscaler_x2,
            upscaler_x4,
            bert_integrater_l2,
            bert_integrater_l1
        ) = initialize_baselayers(config, basemod, tokenizer, stack_id=0)

        self.get_extended_attention_mask = basemod.get_extended_attention_mask
        self.embedding = layer_embedding
        self.layer_basetransformer = layer_basetransformer
        self.maxpool = maxpool
        self.maxpool_attn = maxpool_attn
        self.bert_reducer_l2 = bert_reducer_l2
        self.bert_reducer_l3 = bert_reducer_l3
        self.bert_encoder_midres = bert_encoder_midres
        self.bert_encoder_lowres = bert_encoder_lowres
        self.upscaler_x2 = upscaler_x2
        self.upscaler_x4 = upscaler_x4
        self.bert_integrater_l2 = bert_integrater_l2
        self.bert_integrater_l1 = bert_integrater_l1
        self.stack_id = 0
        if device is None:
            self.to(basemod.device)
            #print(self.device)
            self.device = basemod.device
        else:
            self.to(device)
            self.device = device

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_l2: Optional[torch.Tensor] = None,
        attention_mask_l3: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False
    ):
        input_shape = input_ids
        past_key_values_length =0 if past_key_values is None else len(past_key_values)

        # extend attention mask
        extended_attention_mask_l1 = self.get_extended_attention_mask(attention_mask, input_shape, self.device)
        # downsample the attention mask to l2 dimension
        if attention_mask_l2 is None:
            attention_mask_l2 = self.maxpool_attn(attention_mask.float())
        extended_attention_mask_l2 = self.get_extended_attention_mask(attention_mask_l2,attention_mask_l2.shape, self.device)
        # downsample the attention mask to l3 dimension
        if attention_mask_l2 is None:
            attention_mask_l3 = self.maxpool_attn(attention_mask_l2.float())
        extended_attention_mask_l3 = self.get_extended_attention_mask(attention_mask_l3,attention_mask_l3.shape, self.device)

        # embed
        embedding_output = self.embedding(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            #input_embeds=None,
            past_key_values_length = past_key_values_length
        )

        # first transformer block (vanilla transformer)
        out_l1 = self.layer_basetransformer(
            hidden_states = embedding_output,
            attention_mask = extended_attention_mask_l1,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions
        )
        hidden_states_l1 = out_l1[0]

        # downsample to sequence 1 to length sequence 2
        hiddens_states_l1_reduced = self.maxpool(hidden_states_l1)

        # reduce dimenion on sequence 2
        out_l2 = self.bert_reducer_l2(
            hidden_states = hiddens_states_l1_reduced,
            attention_mask = extended_attention_mask_l2,
            head_mask=head_mask,
            encoder_hidden_states = hidden_states_l1,
            encoder_attention_mask= extended_attention_mask_l1,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
        )
        hidden_states_l2 = out_l2[0]

        # Vanilla transformers block at mid-resolution (1/2 seq-length)
        out_encoder = self.bert_encoder_midres(
            hidden_states=hidden_states_l2,
            attention_mask=extended_attention_mask_l2,
            head_mask = head_mask,
            return_dict=return_dict
        )
        hidden_states_l2 = out_encoder[0]

        # reduce sequence length (1/4 seq-length)
        hiddens_states_l2_reduced = self.maxpool(hidden_states_l2)

        # reduce dimenion on sequence 2
        out_l3 = self.bert_reducer_l3(
            hidden_states = hiddens_states_l2_reduced,
            attention_mask = extended_attention_mask_l3,
            head_mask=head_mask,
            encoder_hidden_states = hidden_states_l2,
            encoder_attention_mask= extended_attention_mask_l2,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
        )
        hidden_states_l3 = out_l3[0]

        #print(hidden_states_l3.shape)
        #print(extended_attention_mask_l3.shape)
        # BertEncoder at low-res
        out_encoder = self.bert_encoder_lowres(
            hidden_states=hidden_states_l3,
            attention_mask=extended_attention_mask_l3,
            head_mask = head_mask,
            return_dict=return_dict
        )
        hidden_states_l3 = out_encoder[0]

        # upscaling: l3 to l2
        hidden_states_upscaled3to2 = self.upscaler_x2(hidden_states_l3)

        # integrate sequence-2 and upscaled sequence-3
        hidden_states_l2 = self.bert_integrater_l2(
            hidden_states = hidden_states_l2,
            attention_mask = extended_attention_mask_l3,
            head_mask = head_mask,
            keyvalue_hidden_states = hidden_states_l3,
            query_to_concat_hidden_states = hidden_states_upscaled3to2,
            query_attention_mask = attention_mask_l2
        )

        # upscaling: l3/l2 to l1 sequence length
        #hidden_states_upscaled3to1 = self.upscaler_x4(hidden_states_l3)
        hidden_states_upscaled2to1 = self.upscaler_x2(hidden_states_l2)
        #hidden_states_upscaled = torch.cat((
        #    hidden_states_upscaled2to1, hidden_states_upscaled3to1
        #),axis=2)

        # integrate low-resolution information back to original dimension
        hidden_states_l1 = self.bert_integrater_l1(
            hidden_states = hidden_states_l1,
            attention_mask = extended_attention_mask_l2,
            head_mask = head_mask,
            keyvalue_hidden_states = hidden_states_l2,
            query_to_concat_hidden_states = hidden_states_upscaled2to1,
            query_attention_mask = extended_attention_mask_l2
        )
        if not return_dict:
            return (
                (hidden_states_l1, hidden_states_l2, hidden_states_l3),
                (extended_attention_mask_l1, extended_attention_mask_l2, extended_attention_mask_l3),
                (attention_mask, attention_mask_l2, attention_mask_l3)
            )
        return {
            "hidden_states": (hidden_states_l1, hidden_states_l2, hidden_states_l3),
            "extended_attention_masks":(extended_attention_mask_l1, extended_attention_mask_l2, extended_attention_mask_l3),
            "attention_masks":(attention_mask, attention_mask_l2, attention_mask_l3)
        }


class AnathemMidModule(nn.Module):
    """Stack of layers that go full circle form high-res to low-res back to high res"""
    def __init__(
            self,
            config,
            basemod=None,
            tokenizer=None,
            past_key_values_length = None,
            device=None,
            stack_id = 1
        ):
        super().__init__()
        self.config = config

        # initalize the layers
        (
            maxpool,
            maxpool_attn,
            bert_reducerintegrator_l2,
            bert_reducerintegrator_l3,
            bert_encoder_midres,
            bert_encoder_lowres,
            upscaler_x2,
            upscaler_x4,
            bert_integrater_l2,
            bert_integrater_l1
        ) = initialize_midlayers(config, basemod, tokenizer, stack_id)

        self.get_extended_attention_mask = get_extended_attention_mask
        self.maxpool = maxpool
        self.maxpool_attn = maxpool_attn
        self.bert_reducerintegrator_l2 = bert_reducerintegrator_l2
        self.bert_reducerintegrator_l3 = bert_reducerintegrator_l3
        self.bert_encoder_midres = bert_encoder_midres
        self.bert_encoder_lowres = bert_encoder_lowres
        self.upscaler_x2 = upscaler_x2
        self.upscaler_x4 = upscaler_x4
        self.bert_integrater_l2 = bert_integrater_l2
        self.bert_integrater_l1 = bert_integrater_l1
        if device is None:
            self.to(basemod.device)
            #print(self.device)
            self.device = basemod.device
        else:
            self.to(device)
            self.device = device

    def forward(
        self,
        hidden_states_highres: torch.Tensor,
        hidden_states_midres: torch.Tensor,
        hidden_states_lowres: torch.Tensor,
        attention_mask: Optional[List[torch.FloatTensor]] = None,
        extended_attention_mask_highres: Optional[List[torch.FloatTensor]] = None,
        extended_attention_mask_midres: Optional[List[torch.FloatTensor]] = None,
        extended_attention_mask_lowres: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False
    ):
        input_shape = hidden_states_highres.shape[:2]
        past_key_values_length =0 if past_key_values is None else len(past_key_values)

        # extend attention mask
        if extended_attention_mask_highres is None:
            extended_attention_mask_highres = self.get_extended_attention_mask(attention_mask, input_shape, self.device)
        if extended_attention_mask_midres is None:
            attention_mask_midres = self.maxpool_attn(attention_mask.float())
            extended_attention_mask_midres = self.get_extended_attention_mask(attention_mask_midres,attention_mask_midres.shape, self.device)
        if extended_attention_mask_lowres is None:
           attention_mask_lowres = self.maxpool_attn(attention_mask_midres.float())
           extended_attention_mask_lowres = self.get_extended_attention_mask(attention_mask_lowres,attention_mask_lowres.shape, self.device)

        # downsample to sequence 1 to length sequence 2
        hiddens_states_l1_reduced = self.maxpool(hidden_states_highres)

        # reduce dimenion on sequence 2
        hidden_states_l2 = self.bert_reducerintegrator_l2(
            inputs = hidden_states_highres, # from highres outputs previous layer (key, values)
            hidden_states = hidden_states_midres, # previous hidden-states for skip connection (short squence-dim, low-res)
            attention_mask = extended_attention_mask_midres,
            head_mask=None,
            query_hidden_states = hiddens_states_l1_reduced
        )

        # Vanilla transformers at mid-resolution (1/2 sequence-length)
        out_encoder = self.bert_encoder_midres(
            hidden_states=hidden_states_l2,
            attention_mask=extended_attention_mask_midres,
            head_mask = None,
            return_dict=return_dict
        )
        hidden_states_l2 = out_encoder[0]

        # reduce sequence length (to 1/4 sequence-length)
        hiddens_states_l2_reduced = self.maxpool(hidden_states_l2)

        # reduce dimenion on sequence 2
        hidden_states_l3 = self.bert_reducerintegrator_l3(
            inputs = hidden_states_midres, # from highres outputs previous layer (key, values)
            hidden_states = hidden_states_lowres, # previous hidden-states for skip connection (short squence-dim, low-res)
            attention_mask = extended_attention_mask_lowres,
            head_mask=None,
            query_hidden_states = hiddens_states_l2_reduced
        )

        # BertEncoder at low-res
        out_encoder = self.bert_encoder_lowres(
            hidden_states=hidden_states_l3,
            attention_mask=extended_attention_mask_lowres,
            head_mask = None,
            return_dict=return_dict
        )
        hidden_states_lowres = out_encoder[0]

        # upscaling: l3 to l2
        hidden_states_upscaled3to2 = self.upscaler_x2(hidden_states_lowres)

        # integrate sequence-2 and upscaled sequence-3
        hidden_states_midres = self.bert_integrater_l2(
            hidden_states = hidden_states_l2,
            attention_mask = extended_attention_mask_lowres,
            head_mask = None,
            keyvalue_hidden_states = hidden_states_lowres,
            query_to_concat_hidden_states = hidden_states_upscaled3to2
        )
        #hidden_states_midres = self.bert_integrative_layer_2(
        #    hidden_states = hidden_states_l2,
        #    attention_mask = extended_attention_mask_midres,
        #    head_mask = None,
        #    query_hidden_states = hidden_states_upscaled3to2)

        # upscaling: l3/l2 to l1 sequence length
        #hidden_states_upscaled3to1 = self.upscaler_x4(hidden_states_lowres)
        hidden_states_upscaled2to1 = self.upscaler_x2(hidden_states_midres)
        #hidden_states_upscaled = torch.cat((hidden_states_upscaled2to1, hidden_states_upscaled3to1),axis=2)

        # integrate low-resolution information back to original dimension
        hidden_states_highres = self.bert_integrater_l1(
            hidden_states = hidden_states_highres,
            attention_mask = extended_attention_mask_midres,
            head_mask = None,
            keyvalue_hidden_states = hidden_states_midres,
            query_to_concat_hidden_states = hidden_states_upscaled2to1
        )

        if not return_dict:
            return (
                (hidden_states_highres, hidden_states_midres, hidden_states_lowres),
                (extended_attention_mask_highres, extended_attention_mask_midres, extended_attention_mask_lowres)
            )
        return {
            "hidden_states": (hidden_states_highres, hidden_states_midres, hidden_states_lowres),
            "attention":(extended_attention_mask_highres, extended_attention_mask_midres, extended_attention_mask_lowres)
        }


class AnathemEncoder(nn.Module):
    """Anathem cores stacks of layers, from embeddings to final transformer block"""
    def __init__(
            self,
            config,
            basemod=None,
            tokenizer=None,
            past_key_values_length = None,
            device=None,
        ):
        super().__init__()
        self.config = config
        self.device = device

        # initialize embedings and first stack
        self.anathem_base_stack = AnathemBaseModule(
            config,
            basemod,
            tokenizer,
            past_key_values_length,
            device,
        )

        # initialize all subsequence stacks
        self.anathem_mid_stack = nn.ModuleList([
            AnathemMidModule(
                config,
                basemod,
                tokenizer,
                past_key_values_length,
                device,
                stack_id = i
            ) for i in range(1, self.config.num_transformer_stacks)
        ])

        # initialize the final transformer modules
        self.final_transformer_block = initialize_finaltransformerlayers(
            config,
            basemod,
            tokenizer,
            stack_id=self.config.num_transformer_stacks+1
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_l2: Optional[torch.Tensor] = None,
        attention_mask_l3: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False
    ):

        # embed and run through first stack of transformers
        hidden_states, extended_attention_masks, attention_masks = self.anathem_base_stack(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_l2=attention_mask_l2,
            attention_mask_l3=attention_mask_l3,
            token_type_ids=token_type_ids, #: Optional[torch.Tensor] = None,
            position_ids=position_ids,#: Optional[torch.Tensor] = None,
            head_mask=head_mask,#: Optional[torch.Tensor] = None,
            inputs_embeds=None,#: Optional[torch.Tensor] = None,
            encoder_hidden_states=None,#: Optional[torch.Tensor] = None,
            encoder_attention_mask=None,#: Optional[torch.Tensor] = None,
            past_key_values=past_key_values,#: Optional[List[torch.FloatTensor]] = None,
            use_cache=use_cache,#: Optional[bool] = None,
            output_attentions=output_attentions,#: Optional[bool] = None,
            output_hidden_states=output_hidden_states,#: Optional[bool] = None,
            return_dict=return_dict
        )

        # middle stack of transformers
        for i, anathem_stack in enumerate(self.anathem_mid_stack):

            # run through each stack (1-2)
            hidden_states, extended_attention_masks = anathem_stack(
                hidden_states_highres = hidden_states[0],
                hidden_states_midres = hidden_states[1],
                hidden_states_lowres = hidden_states[2],
                extended_attention_mask_highres = extended_attention_masks[0],
                extended_attention_mask_midres = extended_attention_masks[1],
                extended_attention_mask_lowres = extended_attention_masks[2]
            )

        # hidden states (high,med,low resolution)
        hidden_states_highres, hidden_states_midres, hidden_states_lowres = hidden_states

        # run through final transformer block (pretrained)
        out_final = self.final_transformer_block(
            hidden_states = hidden_states_highres,
            attention_mask = extended_attention_masks[0],
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions
        )
        #print(type(out_final))
        #print(len(out_final))
        hidden_states_highres = out_final[0]
        if not output_attentions:
            return (hidden_states_highres, hidden_states_midres, hidden_states_lowres), attention_masks

        attention_final = out_final[1]
        return (hidden_states_highres, hidden_states_midres, hidden_states_lowres), attention_masks, attention_final


class BertGenericClassificationHead(nn.Module):
    """Instantiates a basic classification head that takes the CLS token and mean of the final layer for classification"""
    def __init__(self, config, n_classes = 1, activation = 'sigmoid', device=None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, n_classes)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = lambda x: x
        if device is not None:
            self.to(device)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        output_vectors=[]
        first_token_tensor = hidden_states[:, 0]
        output_vectors.append(first_token_tensor)
        # mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        # concatenate
        pooled_output = torch.concat(output_vectors, axis=1)
        #print(pooled_output.shape)
        logits = self.dense(pooled_output)
        return self.activation(logits)


class AnathemMultiSiloPooler(nn.Module):
    """
    Pools the token-embeddings along the sequence dimenions for a final sentence-vector.
    The pooling occuras across all three 'silos'
    The pooling consists of the CLS token as well as mean pooling, concatenated token
    Use the pooling outputs prior to any sequenceClassification
    """
    def __init__(
        self,
        config,
        dim_out = None,
        mean_activation = nn.Tanhshrink,
        out_activation = None,
        dims_in = None,
        p_dropout:float=None,
        device=None
    ):
        super().__init__()
        """See the `sequence_classification_out_dim`
        I added do_pair_cls to the config and n_labels
        the AnaTransformer will now have a pair_cls head as well, that doesn't output during forward, but can be used for pair classification


        I am thinking about changing this
        - `sequence_classification_out_dim` should be for label classification, like for another classifier on top of the sentence_embedding_dim
        - I think the nn.Linear should operate on the full concat([cls_tokens, mean_pooled]), rather than individually on the cls_token / mean_tokens
        - Should there be a linear transformation of the final layer before mean-pooling?
        - Should the classification-head operate on `concat([cls_tokens, mean_pooled])` or nn.Linear(concat([cls_tokens, mean_pooled]))?
            - in Bert, there is always another nn.Linear(cls_tokens) before the pooled output is ingested into the classification-head
         """
        # dimensions of the hiddens states being processed as inputs
        if dims_in is None:
            try:
                dims_in = config.sequence_classification_intermediate_dim
            except:
                dims_in = [dim_out, dim_out//2, dim_out//4]
        self.dims_in = dims_in
        self.dim_in = sum(dims_in)
        self.hidden_size = config.hidden_size
        if dim_out is None:
            try:
                dim_out = config.sequence_classification_out_dim
            except:
                dim_out = config.hidden_size*2
        self.dim_out = dim_out
        self.mean_activation = mean_activation

        #self.dense = nn.Linear(config.hidden_size*2, n_classes)
        if out_activation == 'none' or out_activation is None:
            self.activation = lambda x: x
        elif out_activation == 'tanh':
            self.activation = nn.Tanh()
        elif out_activation == 'relu':
            self.activation = nn.ReLU()
        elif out_activation == 'sigmoid':
            self.activation = torch.sigmoid

        # before mean-pooling, apply some dropout and TanhShrink
        self.pre_mean_pooling = nn.Sequential(
            nn.Dropout(p_dropout),
            self.mean_activation
        )

        # previously I had some dropout and nn.Linear on the CLS
        #self.cls_pooler = nn.Sequential(
        #    nn.Dropout(p_dropout), nn.Linear(self.dim_in, int(self.hidden_size)),
        #)
        # sequential layer to concatenate the mean tokens from multiple tokens
        #self.mean_pooler = nn.Linear(self.dim_in, self.hidden_size)

        # dense layer + activation on the final concatenated pooled output
        self.out_transform = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(self.dim_in*2, self.dim_out),
            self.activation
        )

        if device is not None:
            self.to(device)

    def forward(self, hidden_states, attention_masks, excess_cls_ids=None) -> torch.Tensor:
        """Combines CLS token and mean-pooling for the sentence-vectorization"""

        # CLS/first-tokens from all silos, all concatenated together
        first_token_tensors_per_silo = self._get_cls_tokens_all_silos(hidden_states)

        # mean pooling per silo and concatenate
        mean_pooled_tensors_per_silo = self._mean_pool_all_silos(hidden_states, attention_masks, excess_cls_ids)

        # concatenate CLS and mean
        pooled_output = torch.concat(first_token_tensors_per_silo+mean_pooled_tensors_per_silo, axis=1)

        return self.out_transform(pooled_output)

    def _get_cls_token(self, hidden_state):
        """Grabs the CLS token from a hidden-states"""
        return hidden_state[:, 0]

    def _get_cls_tokens_all_silos(self, hidden_states):
        """Grabs the CLS token from all hidden_states"""
        first_tokens_per_silo = [
            self._get_cls_token(hidden_state) for hidden_state in hidden_states
        ]
        # concat all first tokens
        #all_first_tokens_cat = torch.cat(first_tokens_per_silo,axis=1)
        # run the concatenated first-tokens through Dense
        #all_first_tokens_out = self.cls_pooler(all_first_tokens_cat)
        return first_tokens_per_silo

    def _mean_pool(self, hidden_state, attention_mask=None, excess_cls_id=None):
        """Pool along a sequence dimension (for just one silo)"""
        if excess_cls_id is None:
            excess_cls_id = attention_mask
        input_mask_expanded = excess_cls_id.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _mean_pool_all_silos(self, hidden_states, attention_masks=None, excess_cls_ids=None):
        """Pool along a sequence dimension (for all silos)"""
        if excess_cls_ids is None:
            excess_cls_ids = attention_masks

        # pre-pool: dense-layer before pooling
        shrunken_hidden_states_per_silo = [
            self.pre_mean_pooling(hidden_state) for hidden_state in hidden_states
        ]

        # mean pool each silo
        mean_pooled_states_per_silo = [
            self._mean_pool(
                hidden_state=hidden_state, excess_cls_id=excess_cls_id
            ) for hidden_state, excess_cls_id
            in zip(shrunken_hidden_states_per_silo, excess_cls_ids)
        ]

        # concat all mean-pooled states
        #all_mean_pooled_states = torch.cat(mean_pooled_states,axis=1)
        # run the concatenated meanpooled states through Dense
        #all_mean_pooled_states = self.mean_pooler(all_mean_pooled_states)
        return mean_pooled_states_per_silo



class AnathemMultiSiloPooler_v1(nn.Module):
    """
    Pools the token-embeddings along the sequence dimenions for a final sentence-vector.
    The pooling occuras across all three 'silos'
    The pooling consists of the CLS token as well as mean pooling, concatenated token
    Use the pooling outputs prior to any sequenceClassification
    """
    def __init__(
        self,
        config,
        dim_out = None,
        mean_activation = nn.Tanhshrink,
        out_activation = nn.Tanh,
        dims_in = None,
        p_dropout=None,
        device=None
    ):
        super().__init__()
        raise NotImplementedError('old version not used. This one had a linear layer per CLS & Mean output')

        # dimensions of the hiddens states being processed as inputs
        if dims_in is None:
            try:
                dims_in = config.sequence_classification_intermediate_dim
            except:
                dims_in = [dim_out, dim_out//2, dim_out//4]
        self.dims_in = dims_in
        self.dim_in = sum(dims_in)
        self.hidden_size = config.hidden_size
        if dim_out is None:
            try:
                dim_out = config.sequence_classification_out_dim
            except:
                dim_out = config.hidden_size*2
        self.dim_out = dim_out
        self.mean_activation = mean_activation

        #self.dense = nn.Linear(config.hidden_size*2, n_classes)
        if out_activation == 'none' or out_activation is None:
            self.activation = lambda x: x
        elif out_activation == 'tanh':
            self.activation = nn.Tanh()
        elif out_activation == 'relu':
            self.activation = nn.ReLU()
        elif out_activation == 'sigmoid':
            self.activation = torch.sigmoid

        # before pooling, apply some dropout and TanhShrink
        self.pre_poolers = nn.Sequential(
            nn.Dropout(p_dropout),
            self.mean_activation
        )

        # linear layer operating on the concatenated CLS tokens from all silos
        self.cls_pooler = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(self.dim_in, int(self.hidden_size)),
        )

        # sequential layer to concatenate the mean tokens from multiple tokens
        self.mean_pooler = nn.Linear(self.dim_in, self.hidden_size)

        if device is not None:
            self.to(device)
        self.device = device

    def forward(self, hidden_states, attention_masks, excess_cls_ids=None) -> torch.Tensor:
        """Combines CLS token and mean-pooling for the sentence-vectorization"""

        # CLS/first-tokens from all silos, all concatenated together
        first_token_tensors = self._get_cls_tokens_all_silos(hidden_states)

        # mean pooling
        mean_pooled_tensors = self._mean_pool_all_silos(hidden_states, attention_masks, excess_cls_ids)

        # concatenate CLS and mean
        pooled_output = torch.concat((first_token_tensors, mean_pooled_tensors), axis=1)

        return self.activation(pooled_output)

    def _get_cls_token(self, hidden_state):
        """Grabs the CLS token from a hidden-states"""
        return hidden_state[:, 0]

    def _get_cls_tokens_all_silos(self, hidden_states):
        """Grabs the CLS token from all hidden_states"""
        first_tokens = [
            self._get_cls_token(hidden_state) for hidden_state in hidden_states
        ]
        # concat all first tokens
        all_first_tokens_cat = torch.cat(first_tokens,axis=1)
        # run the concatenated first-tokens through Dense
        all_first_tokens_out = self.cls_pooler(all_first_tokens_cat)
        return all_first_tokens_out

    def _mean_pool(self, hidden_state, attention_mask=None, excess_cls_id=None):
        """Pool along a sequence dimension (for just one silo)"""
        if excess_cls_id is None:
            excess_cls_id = attention_mask
        input_mask_expanded = excess_cls_id.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def _mean_pool_all_silos(self, hidden_states, attention_masks=None, excess_cls_ids=None):
        """Pool along a sequence dimension (for all silos)"""
        if excess_cls_ids is None:
            excess_cls_ids = attention_masks

        # pre-pool: dense-layer before pooling
        hidden_states = [
            self.pre_poolers(hidden_state) for hidden_state in hidden_states
        ]

        # mean pool each silo
        mean_pooled_states = [
            self._mean_pool(
                hidden_state=hidden_state, excess_cls_id=excess_cls_id
            ) for hidden_state, excess_cls_id
            in zip(hidden_states, excess_cls_ids)
        ]

        # concat all mean-pooled states
        all_mean_pooled_states = torch.cat(mean_pooled_states,axis=1)

        # run the concatenated meanpooled states through Dense
        all_mean_pooled_states = self.mean_pooler(all_mean_pooled_states)
        return all_mean_pooled_states


class PairClassificationHead(nn.Module):
    """Bert Attention Layer that uses a dimension-reduced version of the query, so to reduce the dimension of the outputs"""
    def __init__(
        self,
        config:Optional[BertConfig] = None,
        hidden_size:Optional[int] = 512,
        do_subtract:Optional[bool] = True,
        dropout:Optional[float] = 0.1,
        n_labels:Optional[int] = 22,
        device=None
    ):
        """Special type of Bert Self attention that reduces the dimension of the inputs by half"""
        super().__init__()
        if not config is None:
            hidden_size = config.hidden_size
            n_labels = config.n_labels
            dropout = config.hidden_dropout_prob

        self.hidden_size = hidden_size
        self.do_subtract = do_subtract if do_subtract is not None else True
        self.dropout_p = dropout
        self.n_labels = n_labels
        self.size_of_concatenated_inputs = self.hidden_size*2*2 + self.do_subtract*self.hidden_size*2

        # final output
        self.layer = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(self.size_of_concatenated_inputs, self.n_labels)
        )
        if device is not None:
            self.to(device)
        self.device = device

    def forward(self, input1, input2):
        features_concat = torch.concat((
            input1,
            input2,
            torch.sub(input1,input2)
        ),axis=1)
        return self.layer(features_concat)


class AnathemTransformer(nn.Module):
    def __init__(
        self,
        config:Dict[str,Any]|None=None,
        device:device|None=None,
        do_mlm:bool|None = None,
        do_cls:bool|None  = None,
        do_pair_cls:bool|None  = None
    ):
        super().__init__()

        # default config
        if config is None:
            config = make_config()
        self.config = config
        self.do_mlm = config.do_mlm if do_mlm is None else do_mlm
        self.do_cls = config.do_cls if do_cls is None else do_cls
        self.do_pair_cls =config.do_pair_cls if do_pair_cls is None else do_pair_cls

        # device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device= device

        # get the basemodel (and its masked LM head
        self.model_string = self.config.model_string
        basemodelLM_pretrained = AutoModelForMaskedLM.from_pretrained(self.model_string)
        basemodelLM_pretrained.to(self.device)

        # get the Pretrained BertEncoder
        basemod_pretrained = get_to_bertlayer(
            basemodelLM_pretrained,
            target_layer_name = 'encoder'
        )

        # make the tokenizer (based on pretrained)
        self.tokenizer = CustomTokenizer(
            model_string=self.config.model_string,
            n_cls_prepend = int(1/config.scale_ratio3),
            n_pad_to_multiple_of= int(1/config.scale_ratio3)
        )

        # make the Embedding and first layers (pretrained)
        self.encoder = AnathemEncoder(
            self.config,
            basemod=basemod_pretrained,
            tokenizer=self.tokenizer ,
            past_key_values_length = None,
            device=self.device,
        )

        # get the Pretrained maskedLM head
        if self.do_mlm:
            # perform maskedLM
            self.mlm = get_to_bertlayer(
                basemodelLM_pretrained,
                target_layer_name = 'cls'
            )
        else:
            self.mlm = lambda x : x

        # make the sequence-classification head
        if self.do_cls:
            self.pooler = AnathemMultiSiloPooler(
                config=self.config,
                dim_out = self.config.sequence_classification_out_dim,
                out_activation='tanh',
                mean_activation = nn.Tanhshrink(),
                dims_in = self.config.sequence_classification_intermediate_dim,
                p_dropout=self.config.hidden_dropout_prob,
                device=self.device
            )

        if self.do_pair_cls:
            self.pair_classifier = PairClassificationHead(
                config = self.config,
                do_subtract=True,
                device = self.device
            )

    def _get_name(self):
        return 'ANATHEM_MODEL_FOR_MLM'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_l2: Optional[torch.Tensor] = None,
        attention_mask_l3: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        excess_cls_ids: Optional[torch.Tensor] = None,
        excess_cls_ids_l2: Optional[torch.Tensor] = None,
        excess_cls_ids_l3: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False
    ):

        # run through base-layer (embeddings, transformer-block, 1 anathem stack)
        outputs_encoder = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_l2=attention_mask_l2, # optional downsized attention mask for sequence-dim 1/2
            attention_mask_l3=attention_mask_l3, # optional downsized attention mask for sequence-dim 1/4
            token_type_ids=token_type_ids, #: Optional[torch.Tensor] = None,
            position_ids=position_ids,#: Optional[torch.Tensor] = None,
            head_mask=head_mask,#: Optional[torch.Tensor] = None,
            inputs_embeds=None,#: Optional[torch.Tensor] = None,
            encoder_hidden_states=None,#: Optional[torch.Tensor] = None,
            encoder_attention_mask=None,#: Optional[torch.Tensor] = None,
            past_key_values=past_key_values,#: Optional[List[torch.FloatTensor]] = None,
            use_cache=use_cache,#: Optional[bool] = None,
            output_attentions=output_attentions,#: Optional[bool] = None,
            output_hidden_states=output_hidden_states,#: Optional[bool] = None,
            return_dict=False
        )
        if output_attentions:
            hidden_states, extended_attention_masks, attention = outputs_encoder
        else:
            hidden_states, extended_attention_masks = outputs_encoder
            attention = None

        out_mlm = {'logits':None}
        out_pooled_vector = None
        hidden_states_highres, hidden_states_midres, hiddenstates_lowres = hidden_states

        # MLM outputs
        if self.do_mlm:
            out_mlm = self.mlm(hidden_states_highres)

        # sequence pooling (for classification)
        if self.do_cls:
            out_pooled_vector = self.pooler(
                hidden_states=hidden_states,
                attention_masks=(attention_mask, attention_mask_l2, attention_mask_l3),
                excess_cls_ids=(excess_cls_ids, excess_cls_ids_l2, excess_cls_ids_l3)
            )
        #
        if return_dict:
            return {
                'hidden_states':(hidden_states_highres, hidden_states_midres, hiddenstates_lowres),
                'pooled':out_pooled_vector,
                'logits':out_mlm['logits'],
                'attention':attention,
                'extended_attention_masks':extended_attention_masks
            }
        return hidden_states, out_pooled_vector, out_mlm, attention, extended_attention_masks


