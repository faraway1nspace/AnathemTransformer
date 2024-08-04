import json
import os
import sys
import torch
from torch.cuda import is_available
if is_available():
    target_device = torch.device('cuda')
else:
    target_device = torch.device('cpu')

from src.configs.constants import MAX_SEQ_LENGTH,SEED
from src.configs.training_config import config_training
from src.configs.model_anathem_config import config_model_anathem
from src.model.anathem_transformer import AnathemTransformer, make_config_anathem
from src.model.model_utils import batch_to_device
from src.model.teachers import TeacherEmbedder
from src.training.batching import DataCollatorForWholeWordMaskAnamod
from src.training.experiment_tracker import ExperimentTracker
from src.training.losses import LossWeight
from src.training.tasks import (
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
)->AnathemTransformer:
    """Initializes the Anathem Transformer."""
    pass


