import json
import os
import sys
import torch
from torch import nn
from torch.cuda import is_available
from torch.optim import Optimizer,AdamW
from torch.optim.lr_scheduler import CyclicLR,LRScheduler
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForMaskedLM
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.configuration_bert import BertConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from torch.utils.data import Dataset
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable


from src.configs.constants import MAX_SEQ_LENGTH,SEED
from src.configs.training_config import config_training as default_config_training
from src.configs.model_anathem_config import config_model_anathem as default_config_model
from src.data_utils.run import load_data
from src.model.anathem_transformer import AnathemTransformer, make_config_anathem
from src.model.tokenizers import CustomTokenizer
from src.model.model_utils import batch_to_device
from src.model.teachers import TeacherEmbedder
from src.training.batching import DataCollatorForWholeWordMaskAnamod
from src.training.experiment_tracker import ExperimentTracker
from src.training.losses import LossWeight, multtask_loss
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
    stat_monitor:List[str]='loss_multitask',
    config_hash:str|None=None
)->ExperimentTracker:
    """Initializes the ExperimentTracker to monitor losses and gradient descent."""
    
    # make the experiment tracker (and possibly reload from existing)
    experiment = ExperimentTracker(
        name=experiment_name,
        config_training=config_training,
        config_model=config_model,
        dir_to_experiments=config_training['dir_to_experiments'],  # where to save checkpoints
        filename_log=filename_log, # where to save experiment logs
        n_steps_patience=config_training.get("steps_patience",20),
        n_steps_eval=config_training.get("eval_steps",500),
        n_steps_checkpoint=config_training.get("checkpoint_steps",75),
        config_hash=config_hash,
    )

    # declare which losses to monitor
    experiment.declare_stats(stats_names)

    # declare which loss/stat will be the early-stopping monitoring state
    experiment.declare_monitor_stat(
        stat_monitor, 
        'minimize', 
        n_steps_patience=config_training.get("steps_patience",20)
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

    # use a cyclic learning rate with exp decrease in max lr
    scheduler = CyclicLR(
        optimizer, 
        base_lr=config_training['lr'], 
        max_lr=config_training['lr']*config_training['lr_max_mult'],
        step_size_up=200, 
        step_size_down=1000, 
        mode="exp_range",
        gamma=0.99967,
        last_epoch=-1 # note this is from the STEP
    )

    return optimizer, scheduler


def initialize_mlmcollator_anathem(
    config_training:Dict[str,Any],
    tokenizer:Union[PreTrainedTokenizer,CustomTokenizer],
):
    """Initializes the MLM Collator for Anathem model."""

    # make collator for MLM task
    custom_mlm_collate_fn = DataCollatorForWholeWordMaskAnamod(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=config_training['mlm_probability']
    )
    
    return custom_mlm_collate_fn

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

        mlm_collator_fn = initialize_mlmcollator_anathem(config_training, tokenizer)
        
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
            collate_fn=mlm_collator_fn,
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


def train_one_epoch_anathem(
    experiment_name:str,
    config_hash:str|None=None, # optional ability to fix the hash
    config_training:Dict[str,Any]|None=None,
    config_model:Dict[str,Any]|None=None,
    filename_log:str = "experiment_log.json",
    target_device:torch.device|None = None,
):
    """Main function: Train the Anathem Transformer on one epoch."""

    # defaults: 
    if config_training is None:
        config_training = default_config_training
    
    if config_model is None:
        config_model = default_config_model

    if target_device is None:
        if is_available():
            target_device = torch.device('cuda')
        else:
            target_device = torch.device('cpu')

    # initialize the BertConfig from the config dictionary
    config_model = make_config_anathem(**default_config_model)

    # initialize the ExperimentTracker
    experiment = initialize_experiment(
        experiment_name,
        config_training,
        config_model,
        filename_log,
        stats_names = ['mlm_loss', 'mlm_distil', 'emb_distil', 'cls_loss',"qa_loss","sts_loss","sts_distil"],
        stat_monitor='loss_multitask',
        config_hash=config_hash
    )

    # get the previous epoch, and other steps
    epoch = experiment.get_epoch()
    print(f"Checkpoints to be saved at: {str(experiment.dir_to_checkpoints)}")

    # initialize the model
    model = initialize_anathem_model(
        config_training,
        config_model,
        target_device
    )

    # initialze the optimizer for model
    optimizer, scheduler = initialize_optimizer(
        model, config_training
    )

    # reload the cached data - multitask, train/val split
    data = load_data(epoch = epoch)

    # initialize the tasks for multitask training
    tasks, weights = initialize_multitasks(
        data,
        model,
        model.tokenizer,
        epoch,
        config_training,
        target_device
    )

    # collect all the tasks into a sequence of tasks for ONE training step
    multi_tasks = [tasks['mlm'],tasks['cls'], tasks['qa'], tasks['mlm'],tasks['cls'], tasks['sts']]

    # collect all the tasks into a sequence of tasks for ONE evaluation step
    eval_tasks = [tasks['mlm'],tasks['cls'], tasks['qa'], tasks['sts']]

    # declare the maximum number of steps for this epoch
    experiment.declare_max_steps(multi_tasks)

    # reset steps for tracking gradient descent
    step_epoch = -1
    if experiment.is_run_reloaded:
        for task in multi_tasks:
            task.set_to_step(start=step_epoch, end=experiment.current_step)
        
        step_epoch = experiment.current_step
        print(f"Continue training at EPOCH: {epoch}; STEP: {experiment.current_step}; GLOBAL STEP: {experiment.global_step}")
        model, optimizer, scheduler, weights = experiment.load_checkpoint(
            model, optimizer, scheduler, weights, method="latest"
        )
        print('Done reloading checkpoints')
        print(experiment.best_stat)
        print(experiment.best_stat_step)
    else:
        print(f"Fresh training at EPOCH: {epoch}; STEP: {experiment.current_step}; GLOBAL STEP: {experiment.global_step}")

    # run gradient descent
    while experiment.do_continue():
        step_epoch += 1
        #print('STEP START %d' % step_epoch)

        losses = {} # 
        for task in multi_tasks:
            optimizer.zero_grad()
            loss = task.step() # calc loss
            loss.backward()
            clip_grad_norm_(task.model.parameters(), config_training['max_grad_norm'])
            optimizer.step()
            losses.update(task.last_loss()) # log the losses
        
        # evaluation and/or checkpointing
        do_eval, do_checkpt = experiment.do_eval(), experiment.do_checkpoint()
        if do_eval or do_checkpt:
            if do_eval:
                for eval_task in eval_tasks:
                    eval_loss = eval_task.evaluate()
                    losses.update(eval_task.last_loss())
                # log the multitask losse
                losses.update({"loss_multitask":multtask_loss(losses)})
                model.train()
            
            experiment.log(step=step_epoch, epoch=epoch, stats=losses)
            # save the checkpoints
            experiment.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                weights=weights,
                method="best" if (do_eval and experiment.do_save_checkpoint()) else "latest"
            )

            print('Saved checkpoints: %s' % str(experiment.latest_checkpoint))
            # save the loss history as csv file
            experiment.save()
        
        print(("%d:" % step_epoch) + "; ".join("%s:%0.4f" % (nm,v) for nm,v in losses.items()))
        # accounting: schedular, weights, experment's global step
        scheduler.step() # decrement lr
        for w in weights:
            w.step() # decrement the weight (less teacher-forcing over time)
        experiment.step() # increment step in the experiments
        #print('STEP END %d' % step_epoch)

    # FINISHED this epoch
    print(f'FINISHED Epoch {epoch}')

    # increment the epoch
    experiment.current_epoch+=1
    # reset the epoch-step
    experiment.current_step=0
    # save the log for next epoch
    experiment.save()

    print('EXITING')
    return experiment







