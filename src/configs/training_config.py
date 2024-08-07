"""Arguments for training and gradient descent."""
from torch.cuda import is_available
from src.configs.constants import MAX_SEQ_LENGTH
config_training = {
    "model_string_teacher_mlm":"bert-base-uncased", # teacher for mlm distribution,
    "model_string_teacher_embedddings":"mixedbread-ai/mxbai-embed-large-v1", # teacher for embeddings
    "batch_size":4 if not is_available() else 10,
    "batch_size_eval":6 if not is_available() else 24,
    "batch_size_qa":3 if not is_available() else 10,
    "lr":0.00004,
    "lr_max_mult":4,
    'optimizer_beta0':0.925,
    "max_seq_length":MAX_SEQ_LENGTH,
    "mlm_probability":0.14,
    "margin_triplet":0.8,
    "max_grad_norm":1.5,
    "weight_distil_mlm_start":0.55,
    "weight_distil_qa_start":0.55,
    "eval_steps":250,
    "checkpoint_steps":100,
    "steps_patience":20,
    "dir_to_experiments":"/content/drive/MyDrive/ScriptsPrograms/ml_anathem_transformer/training/anathem_runs/"
}
