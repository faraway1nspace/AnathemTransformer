import os
from datetime import datetime# import now #output_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
import json
import hashlib
import csv
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
from transformers import BertConfig
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable

from src.training.losses import LossWeight
from src.training.tasks import Task

class ExperimentTracker:
    def __init__(
        self,
        name: str, # name for this experiment
        config_training: Dict[str, Any], # traning configuration for hashing the experiment
        config_model: BertConfig, # model configuration for hashing the experiment
        dir_to_experiments: str,  # where to save checkpoints
        filename_log: str = "experiment_log.json", # where to save experiment logs
        stats_names: List[str]|None = None, # list of names of statistics to monitor (can declare later)
        early_stopping_stat:str|None = None, # name of specific statistic to monitor for early stopping
        optimize: str = "minimize",  # whether to max or minimize the early-stopping statistics
        n_steps_patience: int = 2000, # number of steps to weight for performance to improve before terminating training
        n_steps_eval: int = 500, # number of steps to weight until running evaluation
        max_steps_in_epoch:int|None = None
    ):
        self.name = name.replace(" ","_")
        self.config_training = config_training
        self.config_model = config_model
        # hash the config to check if this is a unique experiment
        self._hash_config()
        # all checkpoint
        self.dir_to_experiment = os.path.join(dir_to_experiments, f"{self.name}_{self.config_hash}")
        os.makedirs(self.dir_to_experiment, exist_ok=True)
        self.dir_to_checkpoints = Path(os.path.join(self.dir_to_experiment, "checkpoints"))
        self.dir_to_checkpoints.mkdir(exist_ok=True)
        self.path_to_checkpoints_model = self.dir_to_checkpoints / "model_weights.pt"
        self.path_to_checkpoints_optimizer = self.dir_to_checkpoints / "optimizer.pt"
        # location of log
        self.path_to_experiment_log = Path(os.path.join(self.dir_to_experiment,filename_log))
        self.csv_path = os.path.join(self.dir_to_experiment,f"history_{self.name}_{self.config_hash}.csv")

        # internal states: stats best stat, etc.
        self.stats = {}
        self.best_stat = None
        self.best_stat_step = -1
        self.early_stopping_stat = early_stopping_stat
        self.optimize = optimize
        self.n_steps_patience = n_steps_patience
        self.n_steps_eval = n_steps_eval
        self.current_step = -1 # step for this epoch (not global step)
        self.current_epoch = 0 # global epoch
        self.global_step = -1
        self.latest_checkpoint = None
        self.checkpoint_every_n_steps = 0 # depreceated
        self.max_steps_in_epoch = max_steps_in_epoch
        # declare the statistics to monitor
        if stats_names:
            self.declare_stats(stats_names)
            if early_stopping_stat and optimize and n_steps_patience:
                self.declare_monitor_stat(early_stopping_stat, optimize, n_steps_patience)
        # set the time
        self.time_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        # declares self to be a fresh-run
        self.is_run_reloaded = False
        # load experiment's saved state (if it exists)
        self._load_experiment()

    def _hash_config(self):
        """Hashes the configs, for saving unique configurations."""
        config_combined = (
            json.dumps(self.config_training, sort_keys=True) +
            json.dumps({k:v for k,v in self.config_model.to_dict().items() if not isinstance(v,torch.device)}, sort_keys=True)
        )
        self.config_hash = hashlib.md5(config_combined.encode()).hexdigest()

    def _load_experiment(self):
        if self.path_to_experiment_log.exists():
            with open(self.path_to_experiment_log, "r") as file:
                log_data = json.load(file)
                self.stats = log_data.get("stats", {})
                self.best_stat = log_data.get("best_stat", None)
                self.best_stat_step = log_data.get("best_stat_step", -1)
                self.early_stopping_stat = log_data.get("early_stopping_stat", None)
                self.optimize = log_data.get("optimize", "minimize")
                self.n_steps_patience = log_data.get("n_steps_patience", 0)
                self.current_step = log_data.get("current_step", 0)
                self.current_epoch = log_data.get("current_epoch", 0)
                self.global_step = log_data.get("global_step", 0)
                self.latest_checkpoint = log_data.get("latest_checkpoint", None)
                self.checkpoint_every_n_steps = log_data.get("checkpoint_every_n_steps", 0)
                self.time_start = log_data.get("time_start", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
                self.dir_to_checkpoints = Path(log_data.get("dir_to_checkpoints", self.dir_to_checkpoints)) #Path(os.path.join(self.dir_to_experiment, "checkpoints"))
                self.path_to_checkpoints_model = Path(log_data.get("path_to_checkpoints_model", self.path_to_checkpoints_model)) #self.dir_to_checkpoints / "model_weights.pt"
                self.path_to_checkpoints_optimizer = Path(log_data.get("path_to_checkpoints_optimizer", self.path_to_checkpoints_optimizer)) #self.dir_to_checkpoints / "optimizer.pt"
                self.csv_path = log_data.get("csv_path",self.csv_path)
        # check if the current is not 0 (and therefore we are re-running from some point)
        if self.current_step>-1:
            self.is_run_reloaded = True

    def declare_stats(self, stats_names: List[str]):
        """Declare which stats to monitor."""
        for stat in stats_names:
            if stat not in self.stats:
                self.stats[stat] = []

    def declare_monitor_stat(self, stat_name: str, optimize: str = "minimize", n_steps_patience: int = 10):
        self.early_stopping_stat = stat_name
        self.optimize = optimize
        self.n_steps_patience = n_steps_patience
        if optimize == 'minimize':
            self.best_stat = 10**15 if self.best_stat is None else self.best_stat
        elif optimize == 'maximize':
            self.best_stat = -10**15 if self.best_stat is None else self.best_stat
        if stat_name not in self.stats:
            self.stats[stat_name] = []

    def declare_max_steps(self, tasks:list[Task])->int:
        """Takes the median number of batches per task as the max number of steps."""
        self.max_steps_in_epoch = int(round(np.quantile([len(task) for task in tasks],0.5)))
        print('running for a maximum of %d steps' % self.max_steps_in_epoch)
        return self.max_steps_in_epoch

    def log(self, step: int, epoch: int, stats: Dict[str, float]):
        self.current_step = step
        self.current_epoch = epoch
        for stat, value in stats.items():
            if stat in self.stats:
                self.stats[stat].append({"global_step":self.global_step, "step": step, "epoch": epoch, "value": value})
            if (
                (stat == self.early_stopping_stat)
                and
                ((self.optimize == "minimize" and value <= self.best_stat) or (self.optimize == "maximize" and value >= self.best_stat))
            ):
                # current stat is the best
                assert stat == self.early_stopping_stat, f"stat:{stat} with value {value}"
                self.best_stat_step = self.global_step
                self.best_stat = value
        #self._save_log()

    def step(self):
        self.global_step+=1

    def do_eval(self)->bool:
        is_eval_interval = ((self.global_step+1) % self.n_steps_eval)==0
        is_end = bool(self.max_steps_in_epoch) and (self.current_step>=self.max_steps_in_epoch)
        return is_eval_interval or is_end

    def do_continue(self) -> bool:
        # check if we've exceeded the total number of steps in this epoch
        if self.max_steps_in_epoch and (self.current_step > self.max_steps_in_epoch):
            return False
        # check if no early stopping step specified
        if not self.early_stopping_stat or self.early_stopping_stat not in self.stats:
            return True
        recent_stats = [
            entry["value"] for entry in self.stats[self.early_stopping_stat]
        ]
        if not recent_stats:
            return True

        current_stat = recent_stats[-1]
        if self.best_stat is None or (self.optimize == "minimize" and current_stat < self.best_stat) or (self.optimize == "maximize" and current_stat > self.best_stat):
            self.best_stat = current_stat
            self.best_stat_step = self.global_step
            return True
        elif round(((self.global_step - self.best_stat_step))/self.n_steps_eval) > self.n_steps_patience:
            # stop if no improvement in loss for  n_steps_patience
            return False
        return True

    def do_save_checkpoint(self) -> bool:
        return self.global_step == self.best_stat_step

    def write(self):
        #csv_path = self.path_to_experiment_log / f"{self.name}_{self.config_hash}.csv"
        #with open(csv_path, "w", newline="") as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow(["stat_name", "step", "epoch", "value"])
        #    for stat, values in self.stats.items():
        #        for value in values:
        #            writer.writerow([stat, value["step"], value["epoch"], value["value"]])
        dfs = [
            pd.DataFrame(stat_values).rename(columns={'value':stat_nm})
            for stat_nm,stat_values in self.stats.items()
        ]
        if len(dfs)==1:
            df = dfs[0]
        else:
            df = pd.merge(dfs[0],dfs[1],how='inner')
        if len(dfs)>2:
            for df_to_merge in dfs[2:]:
                df = pd.merge(df, df_to_merge, how='inner')
        df.to_csv(self.csv_path,index=False)
        print('wrote log csv to %s' % self.csv_path)

    def _save_log(self):
        with open(self.path_to_experiment_log, "w") as file:
            json.dump({
                "stats": self.stats,
                "best_stat": self.best_stat,
                "best_stat_step": self.best_stat_step,
                "early_stopping_stat": self.early_stopping_stat,
                "optimize": self.optimize,
                "n_steps_patience": self.n_steps_patience,
                "current_step": self.current_step,
                "current_epoch": self.current_epoch,
                "global_step":self.global_step,
                "latest_checkpoint": str(self.latest_checkpoint.absolute()),
                "checkpoint_every_n_steps": self.checkpoint_every_n_steps,
                "time_start":self.time_start,
                "dir_to_checkpoints":str(self.dir_to_checkpoints.absolute()),
                "path_to_checkpoints_model":str(self.path_to_checkpoints_model.absolute()),
                "path_to_checkpoints_optimizer":str(self.path_to_checkpoints_optimizer.absolute()),
                "csv_path":self.csv_path
            }, file, indent=4)

    def save(self)->None:
        """Writes the csv results and saves the log."""
        self._save_log()
        self.write()

    def save_checkpoint(
        self,
        model:nn.Module,
        optimizer:Optimizer,
        scheduler:LRScheduler,
        weights:List[LossWeight]|None = None
    ):
        # save the model weights and some metadata
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'step': self.current_step,
                'epoch': self.current_epoch,
                'global_step':self.global_step,
                'config_training':self.config_training,
                'config_model':self.config_model.to_dict(),
                'name':self.name,
                'time':datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            }, self.path_to_checkpoints_model
        )
        # save the optimizer (and scheduler and distillation weights as well, if provided)
        optimizer_states = {
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': self.current_step,
            'epoch': self.current_epoch,
            'global_step':self.global_step
        }
        if scheduler:
            optimizer_states.update({'scheduler':scheduler.state_dict()})
        if weights:
            optimizer_states.update({'weights':[]})
            for w in weights:
                if isinstance(w,LossWeight):
                    optimizer_states['weights'].append(w.state_dict())
                else:
                    optimizer_states['weights'].append(w)
        torch.save(optimizer_states, self.path_to_checkpoints_optimizer)
        self.latest_checkpoint = self.path_to_checkpoints_model
        self._save_log()

    def load_checkpoint(
        self,
        model:nn.Module,
        optimizer:Optimizer,
        scheduler:LRScheduler|None=None,
        weights:List[LossWeight]|None = None
    )->Tuple[nn.Module, Optimizer, LRScheduler|None, List[LossWeight]|None]:
        if self.path_to_checkpoints_model.exists():
            checkpoint = torch.load(self.path_to_checkpoints_model)
            model.load_state_dict(checkpoint['model_state_dict'])
            #self.current_step = checkpoint['step']
            #self.current_epoch = checkpoint['epoch']
            #self.global_step =  checkpoint['global_step']
            print(f'Loaded checkpoint at Epoch: {self.current_epoch}; step:{self.global_step}')
        else:
            raise FileNotFoundError(f"No checkpoint found at {self.path_to_checkpoints_model}")
        if self.path_to_checkpoints_optimizer.exists():
            checkpoint = torch.load(self.path_to_checkpoints_optimizer)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if bool(checkpoint['weights']):
                if isinstance(checkpoint['weights'][0],dict) and bool(weights) and isinstance(weights[0],LossWeight):
                    for state_dict,w in zip(checkpoint['weights'], weights):
                        w.load_state_dict(state_dict)
                elif isinstance(checkpoint['weights'][0],dict) and bool(weights) and isinstance(weights[0],float):
                    weights = [
                        LossWeight.from_state_dict(state_dict) 
                        for state_dict,w in zip(checkpoint['weights'], weights)
                    ]
                elif isinstance(checkpoint['weights'][0],dict) and not bool(weights):
                    weights = [
                        LossWeight.from_state_dict(w_state_dict) 
                        for w_state_dict,w in zip(checkpoint['weights'], weights)
                    ]
                elif isinstance(checkpoint['weights'][0],float):
                    weights = checkpoint['weights']
                else:
                    raise NotImplementedError('unknown type of saved weight %s' % str(type(checkpoint['weights'][0])))
        return model, optimizer, scheduler, weights

