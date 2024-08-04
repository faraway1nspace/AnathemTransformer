import numpy as np
from torch import nn,Tensor
from torch.nn.functional import relu,cosine_similarity
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable


# MLM distillation loss function (kl-divergence between teacher and student outputs)
loss_fn_mlm_distil = nn.KLDivLoss(reduction="batchmean")

# MLM loss for labels
loss_fn_mlm_labels = nn.CrossEntropyLoss(ignore_index=-100) # non-masked tokens have -100

# classification loss
loss_fn_cls = nn.BCEWithLogitsLoss(reduction='none') # no reduction because we need to mask-out the invalid row/column combinations

# MLM-pooled output distillation loss
loss_fn_mlmpooling_distil = nn.MSELoss()

# QA-task cosine triplet loss
def cosine_triplet_loss(
    anchor:Tensor, pos:Tensor, neg:Tensor, margin_triplet:float = 10.0
):
    """Cosine similarity between two pairs (a-b,a-c) by a margin."""
    distance_pos = 1-cosine_similarity(anchor, pos)
    distance_neg = 1-cosine_similarity(anchor, neg)
    losses = relu(distance_pos - distance_neg + margin_triplet)
    return losses.mean()


def multtask_loss(losses:Dict[str,float])->float:
    """Combine multiple losses into one for early-stopping loss (geo metric mean)."""
    eval_losses_to_combine = np.array([v for k,v in losses.items() if 'eval' in k])
    # Use logarithms to avoid numerical issues
    return np.exp(np.mean(np.log(eval_losses_to_combine)))


class LossWeight:
    """A float for weighting the teacher-loss versus task-loss (w, 1-w), with special method for decrementing by step."""
    def __init__(self, start_value:float, decrement_per_step:float=0.0001, min:float=0.1, current_step:int =0, value:float|None=None):
        self.start_value = start_value
        self.decrement_per_step = decrement_per_step
        self.min = min
        self.current_step = 0
        if value is None:
            self.value = start_value
            if current_step !=0:
                for _ in range(current_step):
                    self.step() # decreases weight and increases current_step
                print('reinitializing weight at step %d with value %0.6f' % (current_step, self.value))
        else:
            self.value = value
            self.current_step = current_step   

    def step(self):
        """Decrements self.value by self.decrement_per_step."""
        self.current_step+=1
        self.value -= self.decrement_per_step
        self.value = max(self.min, self.value)

    def __call__(self):
        return self.value

    def inverse(self):
        return 1-self.value

    def __mul__(self, other):
        if isinstance(other, (int, float, Tensor)):
            return self.value * other
        elif isinstance(other, LossWeight):
            return self.value * other.value
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Weight' and '{type(other).__name__}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float,Tensor)):
            return self.value - other
        elif isinstance(other, LossWeight):
            return self.value - other.value
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Weight' and '{type(other).__name__}'")

    def __rsub__(self, other):
        if isinstance(other, (int, float, Tensor)):
            return other - self.value
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{type(other).__name__}' and 'Weight'")

    def state_dict(self):
        """For saving the internal state of the weight, like other torch methods optimizer.state_dict()."""
        return {
            "start_value":self.start_value,
            "decrement_per_step":self.decrement_per_step,
            "min":self.min,
            "current_step":self.current_step,
            "value":self.value
        }

    def load_state_dict(self, state_dict:Dict[str,Union[int,float]])->None:
        """Reload saved state"""
        for k,v in state_dict.items():
            setattr(self,k,v)

    @classmethod
    def from_state_dict(cls, state_dict:Dict[str,Union[int,float]])->'LossWeight':
        return cls(
            start_value=state_dict["start_value"],
            decrement_per_step=state_dict["decrement_per_step"],
            min=state_dict["min"],
            current_step=state_dict["current_step"],
            value=state_dict["value"]
        )
