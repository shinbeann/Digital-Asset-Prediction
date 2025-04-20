####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import *
from pprint import pformat
import textwrap

# Libs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_training_session_information(
    save_dir: str,
    sequence_size: int, 
    stride: int,
    random_state: int,
    optimizer_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
    model_kwargs: Dict[str, Any],
    test_loss: float, 
    test_mae: float, 
    test_r2: float,
    test_explained_var: float,
):
    info_file = os.path.join(save_dir, "model_info.txt")
    
    model_hparam_text = (
        "Default model hyperparameter values used."
        if not model_kwargs else
        pformat(model_kwargs)
    )
    
    text = textwrap.dedent(f"""\
        [Dataset Configuration]
        Sequence size: {sequence_size}
        Stride: {stride}
        Random seed: {random_state}
        
        [Training Hyperparameters]
        Optimizer: {optimizer_name}
        Batch Size: {batch_size}
        Learning Rate: {learning_rate}
        Epochs: {num_epochs}
        Weight Decay (L2 Regularization): {weight_decay}
        
        [Model Hyperparameters]
        model_kwargs: {model_hparam_text}
        
        [Test Results]
        (Testing model parameters from last epoch)
        Loss: {test_loss}
        MAE: {test_mae}
        R2: {test_r2}
        Explained Variance: {test_explained_var}
    """)
    
    with open(info_file, "w") as f:
        f.write(text)

    print(f"✅ Saved training session information to {info_file}") # I love emojis