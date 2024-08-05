import json
import os
import sys
import torch
from torch import nn
from torch.cuda import is_available
from torch.optim import Optimizer,AdamW
from torch.optim.lr_scheduler import CyclicLR,LRScheduler
from transformers import AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset

from src.configs.constants import MAX_SEQ_LENGTH,SEED
from src.configs.training_config import config_training
from src.configs.model_anathem_config import config_model_anathem
from src.model.anathem_transformer import AnathemTransformer, make_config_anathem
from src.model.tokenizers import CustomTokenizer
from src.model.model_utils import batch_to_device
from src.model.teachers import TeacherEmbedder
from src.training.batching import DataCollatorForWholeWordMaskAnamod
from src.training.experiment_tracker import ExperimentTracker
from src.training.losses import LossWeight
from src.training.tasks import (
    Task,
    AnathemTaskMLM,
    AnathemTaskTriplet,
    AnathemTaskPairClassification
)


def initialize_experiment(
    experiment_name:str,
    config_training:Dict[str,Any],
    config_model:Union[PretrainedConfig, BertConfig],
    filename_log:str = "experiment_log.json",
    stats_names:List[str] = ['mlm_loss', 'mlm_distil', 'emb_distil', 'cls_loss',"qa_loss","sts_loss","sts_distil"],
    stat_monitor:List[str]='loss_multitask'
)->ExperimentTracker:
    """Initializes the ExperimentTracker to monitor losses and gradient descent."""
    
    # make the experiment tracker (and possibly reload from existing)
    experiment = ExperimentTracker(
        name=experiment_name,
        config_training=config_training,
        config_model=config_model,
        dir_to_experiments=config_training['dir_to_experiments'],  # where to save checkpoints
        filename_log=filename_log, # where to save experiment logs
        n_steps_patience=config_training.get("steps_patience",10),
        n_steps_eval=config_training.get("eval_steps",3)
    )

    # declare which losses to monitor
    experiment.declare_stats(stats_names)

    # declare which loss/stat will be the early-stopping monitoring state
    experiment.declare_monitor_stat(
        stat_monitor, 
        'minimize', 
        n_steps_patience=config_training.get("steps_patience",10)
    )

    return experiment


def initialize_anathem_model(
    config_training:Dict[str,Any],
    config_model:Union[PretrainedConfig, BertConfig],
    target_device:torch.device|None = None
) -> AnathemTransformer:
    """Initializes the Anathem Transformer."""
    
    anamod = AnathemTransformer(
        config = config_model,
        device = target_device,
        do_mlm = config_model.do_mlm,
        do_cls = config_model.do_cls,
        do_pair_cls = config_model.do_pair_cls
    )

    anamod.train()

    return anamod


def initialize_optimizer(
    model:nn.Module,
    config_training:Dict[str,Any],
) -> Tuple[Optimizer, LRScheduler]:
    """Initializes the optimizer and scheduler."""

    optimizer = AdamW(
        model.parameters(), 
        lr = config_training['lr'], 
        betas=(config_training['optimizer_beta0'], 0.999)
    )

    scheduler = CyclicLR(
        optimizer, 
        base_lr=config_training['lr'], 
        max_lr=config_training['lr']*config_training['lr_max_mult'],
        step_size_up=150, 
        step_size_down=500, 
        mode="exp_range",
        gamma=0.9995,
        last_epoch=-1 # note this is from the STEP
    )

    return optimizer, scheduler


def initialize_multitasks(
    data:Dict[str,Dict[str,Dataset]],
    model:nn.Module,
    tokenizer:Union[PreTrainedTokenizer,CustomTokenizer],
    epoch:int,
    config_training:Dict[str,Any],
    target_device:torch.device
) -> Tuple[Dict[str, Task], List[LossWeight]]:
    """Initializes the tasks."""

    # containers for class to return
    tasks:Dict[str,Task] = {}
    weights:List[LossWeight] = []

    # initialize the teacher for MLM distribution
    if config_training.get('model_string_teacher_mlm',False):

        # download the MLM teacher from huggingface
        teacher_mlm = AutoModelForMaskedLM.from_pretrained(
            config_training['model_string_teacher_mlm']
        )

        # set the teacher mlm device
        teacher_mlm.to(target_device)
        teacher_mlm.eval()

        # initialize the distillation weight
        weight_mlm_distil = LossWeight(config_training['weight_distil_mlm_start'])
        weights.append(weight_mlm_distil)


    # initialize the teacher for embedding tasks
    if config_training.get('model_string_teacher_embedddings',False):

        # download the embedding teacher from huggingface
        teacher_emb = TeacherEmbedder(
            pretrained_name = config_training['model_string_teacher_embedddings'], 
            pooling='cls', 
            device=target_device
        )

        # set teacher to device
        teacher_emb.to(target_device)
        teacher_emb.eval()

        # initialize the distillation weight
        weight_qa_distil = LossWeight(config_training['weight_distil_qa_start'])
        weights.append(weight_qa_distil)

    # mlm task
    if 'mlm' in data['train'].keys():

        # make collator for MLM task
        custom_mlm_collate_fn = DataCollatorForWholeWordMaskAnamod(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=config_training['mlm_probability']
        )
        
        task_mlm = AnathemTaskMLM(
            model,
            data['train']['mlm'],
            data['val']['mlm'],
            batch_size=config_training['batch_size'],
            batch_size_val=config_training['batch_size_eval'],
            weight=weight_mlm_distil,
            config=config_training,
            seed_epoch=SEED+epoch,
            seed_global=SEED,
            teachers={
              "mlm":teacher_mlm,
              "emb":teacher_emb,
            },
            collate_fn=custom_mlm_collate_fn,
            name="mlm",
            device=target_device
        )

        tasks['mlm'] = task_mlm
    
    # cls task (classification)
    if 'cls' in data['train'].keys():

        # make collator for MLM task
        task_cls = AnathemTaskPairClassification(
            model,
            data['train']['cls'],
            data['val']['cls'],
            batch_size=config_training['batch_size'],
            batch_size_val=config_training['batch_size_eval'],
            weight=LossWeight(1.0),
            config=config_training,
            seed_epoch=SEED+epoch,
            seed_global=SEED,
            teachers={}, # no teacher forcing for classification
            collate_fn=None,
            name="cls",
            device=target_device
        )
        tasks['cls'] = task_cls

    if 'qa' in data['train'].keys():

        # make collator for MLM task
        task_qa = AnathemTaskTriplet(
            model,
            data['train']['qa'],
            data['val']['qa'],
            teacher_query_prepend=teacher_emb.query_prefix,
            batch_size=config_training['batch_size_qa'],
            batch_size_val=config_training['batch_size_eval'],
            weight=weight_qa_distil,
            config=config_training,
            seed_epoch=SEED+epoch,
            seed_global=SEED,
            teachers={'emb':teacher_emb},
            collate_fn=None,
            name="qa",
            device=target_device
        )
        tasks['qa'] = task_qa

    if 'sts' in data['train'].keys():

        # make collator for MLM task
        task_sts = AnathemTaskTriplet(
            model,
            data['train']['sts'],
            data['val']['sts'],
            teacher_query_prepend=teacher_emb.passage_prefix,
            batch_size=config_training['batch_size_qa'],
            batch_size_val=config_training['batch_size_eval'],
            weight=weights[1],
            config=config_training,
            seed_epoch=SEED+epoch,
            seed_global=SEED,
            teachers={'emb':teacher_emb},
            collate_fn=None,
            name="sts",
            device=target_device
        )
        tasks['sts'] = task_sts
    
    return tasks, weights



