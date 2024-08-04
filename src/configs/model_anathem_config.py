"""Args for the Anathem transformer model for `make_config` function."""

config_model_anathem = {
    "modelstring":"google/bert_uncased_L-4_H-512_A-8",
    "num_transformer_stacks":3,
    "scale_ratio2":0.5,
    "scale_ratio3":0.25,
    "multiplier_intermediate2":4.0,
    "multiplier_intermediate3":4.0,
    "num_layers_l2":1, # mid-res encoder
    "num_layers_l3":3, # low-res encoder
    "dropout_scaling":0.05,
    "do_cheap_integrator":[1],
    "sequence_classification_intermediate_dim":None, # default is the same as the basemodel hidden-dim
    "sequence_classification_out_dim":None, # default is x2 same as the basemodel hidden-dim
    "do_mlm":True,
    "do_cls":True,
    "do_pair_cls":True,
    "n_labels":24
}