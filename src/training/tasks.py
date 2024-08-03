from src.configs.constants import MAX_SEQ_LENGTH, SEED
from torch import nn, Tensor, device
from torch.utils.data import DataLoader, Dataset
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable

from src.model.model_utils import batch_to_device
from src.training.batching import LengthBatchSampler
from src.training.losses import (
    LossWeight,
    loss_fn_mlm_distil,
    loss_fn_mlm_labels,
    loss_fn_cls,
    loss_fn_mlmpooling_distil,
    cosine_triplet_loss
)

class Task:
    def __init__(
        self,
        model:nn.Module,
        data_train: Dataset,
        data_eval:Dataset,
        batch_size:int,
        batch_size_val:int,
        weight:LossWeight,
        config:dict,
        seed_epoch:int,
        seed_global:int = SEED,
        teachers:Dict[str,nn.Module|None]={},
        collate_fn:Callable|None = None,
        name:str="ml-task",
        device:device|None = None
    ):
        self.name = name
        self.model=model#:nn.Module,
        self.teachers=teachers# = Dict[str,nn.Module],
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.weight = weight
        self.config = config#:dict,
        if device is None:
            device = model.device
        self.target_device = device

        # make the dataloaders
        self.dl_train = self.make_dataloader(
            data_train, batch_size, drop_last=False, collate_fn=collate_fn,dl_sorting_function=self.sort_len, seed = seed_epoch
        )
        self.dl_eval = self.make_dataloader(
            data_eval, batch_size_val, drop_last=False, collate_fn=collate_fn, dl_sorting_function=self.sort_len, seed = seed_epoch
        )
        # iter to step through dataloader of training set
        self.train_iter = iter(self.dl_train)
        # calculate the amount of data
        self.n_batches = len(self.dl_train)
        # for tracking losses
        self.last_loss_log = {}

    def __len__(self)->int:
        return self.n_batches

    @staticmethod
    def sort_len(x:Union[str,Dict[str,str]])->float:
        """Task specific method for sorting batches by text length."""
        return len(x)

    def make_dataloader(
        self,
        data:Dataset,
        batch_size:int,
        drop_last:bool = False,
        collate_fn:Callable|None = None,
        dl_sorting_function:Callable = lambda x:x,
        seed:int = SEED
    )->DataLoader:
        """Initializes a batch-sampler by length of exmaples."""
        return DataLoader(
            data,
            batch_sampler = LengthBatchSampler(
                data,
                batch_size=batch_size,
                drop_last=drop_last,
                get_item_len=dl_sorting_function,
                seed=seed
            ),
            collate_fn=collate_fn
        )

    def next(self)->Dict[str,Any]:
        """Yields one batch from dataloader, including resetting if we hit the end."""
        try:
            batch = next(self.train_iter)
        except StopIteration:
            # reset the dataloader-iter at beginning
            self.train_iter = iter(self.dl_train)
            batch = next(self.train_iter)
        return batch

    def set_to_step(self, start:int, end:int=-1)->None:
        """Increments the dataloader/iter to `end` for continued training at a specific step."""
        counter=start
        while counter<end:
            _ = self.next()
            counter+=1
        print('task %s set to epoch-step %d' % (self.name, counter))

    def _tokenize(self, text_list:List[str])->Dict[str,Union[Tensor, str]]:
        return batch_to_device(
            self.model.tokenizer(
                text_list,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH-4,
                truncation=True,
                padding='longest',
                return_tensors='pt'
            ),self.target_device
        )

    def tokenize(self, batch:Dict[str, Any]):
        """Task-specific wrapper for _tokenize."""
        pass

    def _encode(self, tokens:Dict[str,Tensor])->Dict[str,Tensor]:
        embedded = self.model.forward(
            input_ids = tokens['input_ids'],
            attention_mask = tokens['attention_mask'],
            attention_mask_l2 = tokens['attention_mask_l2'],
            attention_mask_l3 = tokens['attention_mask_l3'],
            excess_cls_ids = tokens['excess_cls_ids'],
            excess_cls_ids_l2 = tokens['excess_cls_ids_l2'],
            excess_cls_ids_l3 = tokens ['excess_cls_ids_l3']
        )
        return embedded

    def encode(self, tokens:Dict[str,Tensor], batch:Dict[str,Any]):
        """Task specific wrapper for _encode."""
        pass

    def calc_loss(self, embed:Dict[str,Tensor], tokens:Dict[str,Tensor], batch:Dict[str,Any]):
        pass

    def calc_teacher_loss(self, embed:Dict[str,Tensor], tokens:Dict[str,Tensor], batch:Dict[str,Any]):
        """Optional calculate distillation loss"""
        pass

    def step(self):
        """Does one complete round of data-pulling & loss calculation."""

        # get data for this step
        batch = self.next()

        # yeild tokens
        tokens = self.tokenize(batch)

        # embed, either sentence/paragraph embeddings or token-embeddings
        embed = self.encode(tokens, batch)

        # calculate loss
        loss_out = self.calc_loss(embed, tokens, batch)
        self.last_loss_log[f"{self.name}_loss"]=loss_out.cpu().detach().item()

        if self.teachers:
            loss_teacher = self.calc_teacher_loss(embed, tokens, batch)
            return self.weight*loss_teacher + self.weight.inverse()*loss_out
        return loss_out

    def evaluate(self,limit:int|None=None):
        pass

    def last_loss(self):
        return self.last_loss_log


class AnathemTaskMLM(Task):
    """A MLM task for the Anathem model to learn masked token IDs, including teacher forcing."""
    def __init__(self, *args, **kwargs):
        # Call the parent class's __init__ method with all arguments
        super().__init__(*args, **kwargs)

    def tokenize(self, batch:Dict[str, Any])->tuple:
        """For the mlm task, the collator is already tokenizing the data so we just return it on the proper device"""
        batch = batch_to_device(batch, self.target_device)
        return batch

    def encode(self, tokens, batch)->Dict[str,Union[Tensor,str]]:
        """Input is simply one tokenized-text by MLM collator, so we merely call ._encode."""
        return self._encode(tokens)

    def calc_loss(
        self,
        embed:Dict[str,Tensor],
        tokens:Dict[str,Tensor],
        batch:Dict[str,Any]
    ):
        loss_mlm_labels = loss_fn_mlm_labels(
            embed[2].view(-1, self.model.config.vocab_size),
            batch['labels'].view(-1)
        )
        return loss_mlm_labels

    def calc_teacher_loss(
        self,
        embed_student:Dict[str,Tensor],
        tokens:Dict[str,Tensor],
        batch:Dict[str,Any]
    ):
        # hidden_states, out_pooled_vector, out_mlm, attention, extended_attention_masks
        distillation_temperature=1.5 # higher values soften the peak of distribution
        mlm_student = embed_student[2]
        with torch.no_grad():
            # mlm teacher outputs
            mlm_teacher = self.teachers['mlm'](
                input_ids = batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # to do this, I'd need to have the original text, and NOT pre-tokenized text -> ha, with my custom collator, I do have the text
            embed_teacher= self.teachers['emb'](
                input_text=batch['text'], prepend_type = 'pos'
            )
            # assert same lengths
            assert mlm_student.size() == mlm_teacher.logits.size()

            # Soften probabilities and compute distillation loss
            loss_mlm_distil = loss_fn_mlm_distil(
                    F.log_softmax(mlm_student / distillation_temperature, dim=-1),
                    F.softmax(mlm_teacher.logits / distillation_temperature, dim=-1)
            ) * (distillation_temperature ** 2)

            # calculate loss between embedding vs. teacher embedding
            loss_mlm_emb = loss_fn_mlmpooling_distil(embed_student[1], embed_teacher)
            self.last_loss_log['mlm_distil']=loss_mlm_distil.cpu().detach().item()
            self.last_loss_log['emb_distil']=loss_mlm_emb.cpu().detach().item()

        # return MLM label loss and distilloss for backprop
        return loss_mlm_distil + loss_mlm_emb

    def evaluate(self,limit:int|None=None):
        """Calculates MLM loss on evaluatoin set."""
        self.model.eval()
        if limit is None:
            limit = len(self.dl_eval)*2
        with torch.no_grad():
            losses = []
            # loop through eval dataloader
            for j,batch in enumerate(self.dl_eval):
                if j>limit:
                    continue
                batch = batch_to_device(batch, self.target_device)
                # get embeddings
                outputs = self.encode(batch,None)
                # get MLM-label loss on val set
                loss_mlm_labels = loss_fn_mlm_labels(
                    outputs[2].view(-1, self.model.config.vocab_size),
                    batch['labels'].view(-1)
                )
                losses.append(loss_mlm_labels.cpu().detach().item())
            eval_loss = sum(losses)/len(losses)
            self.last_loss_log[f"{self.name}_eval"]=eval_loss
            return eval_loss


class AnathemTaskPairClassification(Task):
    """A MLM task for the Anathem model to learn masked token IDs, including teacher forcing."""
    def __init__(self, *args, **kwargs):
        # Call the parent class's __init__ method with all arguments
        self.TEXT_NAMES = ['text1','text2']
        super().__init__(*args, **kwargs)

    def sort_len(self, x:Dict[str,Any])-> float:
        """Sort pairs by length of pair1 and then pair2."""
        return (
            len(x[self.TEXT_NAMES[0]].split(" "))/1.2 + len(x[self.TEXT_NAMES[1]].split(" "))/(MAX_SEQ_LENGTH*TOKEN_FUDGE_FACTOR)
        )

    def tokenize(self, batch:Dict[str, Any])->Dict[str, Dict[str,Any]]:
        tokens_cls = {
            text_column:self._tokenize(batch[text_column])
            for text_column in self.TEXT_NAMES
        }
        return tokens_cls

    def encode(self, tokens, batch)->Tensor:
        # get paragraph embeddings for the cls text 1 and text 2
        pooled_cls = {
            k:self._encode(token_pair)[1]
            for k,token_pair in tokens.items()
        }
        # run through the pair-classification head
        logits = self.model.pair_classifier(
            pooled_cls[self.TEXT_NAMES[0]],
            pooled_cls[self.TEXT_NAMES[1]]
        )
        return logits

    def calc_loss(
        self,
        embed:Tensor,
        tokens:Dict[str,Tensor],
        batch:Dict[str,Any]
    ):
        batch = batch_to_device(batch, self.target_device)
        # calculate classification loss and mask out invalid entries
        loss_cls_masked = loss_fn_cls(embed, batch['labelvector'])*batch['mask']
        # normalize
        loss_cls = loss_cls_masked.sum()/batch['mask'].sum()
        return loss_cls

    def evaluate(self,limit:int|None=None):
        """Calculates Classification loss on evaluatoin set."""
        self.model.eval()
        if limit is None:
            limit = len(self.dl_eval)*2
        with torch.no_grad():
            losses = []
            # loop through eval dataloader
            for j,batch in enumerate(self.dl_eval):
                if j>limit:
                    continue
                tokens = self.tokenize(batch)
                # embed and predict
                pred_logits = self.encode(tokens, batch)
                # calculate per-row/column loss and zero-out the invalid combinations before backprop
                loss_cls = self.calc_loss(pred_logits, tokens, batch)
                # collect losses
                losses.append(loss_cls.cpu().detach().item())

            eval_loss = sum(losses)/len(losses)
        self.last_loss_log[f"{self.name}_eval"]=eval_loss
        return eval_loss

class AnathemTaskTriplet(Task):
    """A MLM task for the Anathem model to learn masked token IDs, including teacher forcing."""
    def __init__(self, *args, teacher_query_prepend="query",**kwargs):
        # Call the parent class's __init__ method with all arguments
        self.TEXT_NAMES = ['query','pos',"neg"]
        super().__init__(*args, **kwargs)
        self.teacher_query_prepend = teacher_query_prepend

    def sort_len(self, x:Dict[str,str])-> float:
        """Sort pairs by length of triplet1 and then triplet2 and then triplet3."""
        d= (MAX_SEQ_LENGTH*TOKEN_FUDGE_FACTOR)
        lens = [len(x[k].split(" "))/d for k in self.TEXT_NAMES]
        lens_weighted = [w*l/1.2 for w,l in zip([100,10,1],lens)]
        return sum(lens_weighted)

    def tokenize(self, batch:Dict[str, Any])->Dict[str, Dict[str,Any]]:
        tokens_cls = {
            text_column:self._tokenize(batch[text_column])
            for text_column in self.TEXT_NAMES
        }
        return tokens_cls

    def encode(self, tokens, batch)->Tensor:
        # get paragraph embeddings for the cls text 1 and text 2
        pooled_triplet = {
            k:self._encode(token_pair)[1]
            for k,token_pair in tokens.items()
        }
        return pooled_triplet

    def calc_loss(
        self,
        embed:Tensor,
        tokens:Dict[str,Tensor],
        batch:Dict[str,Any]
    ):
        # triplet loss for qa
        loss_triplet = cosine_triplet_loss(
            embed[self.TEXT_NAMES[0]],
            embed[self.TEXT_NAMES[1]],
            embed[self.TEXT_NAMES[2]],
            margin_triplet=self.config['margin_triplet']
        )
        return loss_triplet

    def calc_teacher_loss(self, embed:Dict[str,Tensor], tokens:Dict[str,Tensor], batch:Dict[str,Any]):
        """MSEloss between embeddings from an embedding teacher and the pooled outputs."""
        # next, embed the token with the teacher
        with torch.no_grad():
            # to do this, I'd need to have the original text, and NOT pre-tokenized text -> ha, with my custom collator, I do have the text
            pooled_teacher = {
                k:teacher_emb(
                    input_text=batch[k],
                    prepend_type = k, # query, pos, neg
                ) for k in self.TEXT_NAMES
            }

            loss_triplet_distil = sum([
                loss_fn_mlmpooling_distil(v_student, v_teacher)
                for v_student,v_teacher
                in zip(embed.values(), pooled_teacher.values())
            ])/len(embed)
        self.last_loss_log[f"{self.name}_distil"]=loss_triplet_distil.cpu().detach().item()
        return loss_triplet_distil

    def evaluate(self,limit:int|None=None):
        """Calculates Classification loss on evaluatoin set."""
        self.model.eval()
        if limit is None:
            limit = len(self.dl_eval)*2
        with torch.no_grad():
            losses = []
            # loop through eval dataloader
            for j,batch in enumerate(self.dl_eval):
                if j>limit:
                    continue
                tokens = self.tokenize(batch)
                # embed and predict
                embeds_triplet = self.encode(tokens, batch)
                # calculate per-row/column loss and zero-out the invalid combinations before backprop
                loss_triplet = self.calc_loss(embeds_triplet, tokens, batch)
                # collect losses
                losses.append(loss_triplet.cpu().detach().item())

            eval_loss = sum(losses)/len(losses)
        self.last_loss_log[f"{self.name}_eval"]=eval_loss
        return eval_loss