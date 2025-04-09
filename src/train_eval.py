####################
# Required Modules #
####################

# Generic/Built-in
import os
import time
from typing import *
from datetime import datetime
import matplotlib.pyplot as plt

# Libs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

# Custom
from src.dataset import Normalizer
from src.models import CryptoBaseModel


def train_model(
    model: CryptoBaseModel,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 100,
    base_dir: str = "saved_models",
    save_interval: int = 100,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[List[float], List[float], List[float], List[float], Normalizer]:
    """
    Trains the given model on provided crypto pricing dataset (via dataloaders). Will evaluate the model's performance 
    on the validation set every epoch. Model's parameters are saved after each specified number of epochs in the 
    specified base directory.

    Args:
        model (CryptoBaseModel): Crypto model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_dataloader (DataLoader): Data loader with training dataset.
        validation_dataloader (DataLoader): Data loader with validation dataset.
        num_epochs (int, optional): Number of epochs to train model with. Defaults to 15.
        base_dir (str, optional): Directory where model parameters will be saved to. Defaults to "saved_models".
        save_interval (int, optional): The interval (in epochs) after which the model will be saved, i.e., the model 
            will be saved every x epochs. Defaults to 100.
        verbose (bool, optional): Whether to print the model's validation metrics after each training epoch. 
            Defaults to True.
        device (Optional[torch.device], optional): The device the model and batch data should be loaded on. 
            Defaults to None, in which case the device will be set to CUDA if available, or CPU otherwise.
            
    Returns:
        Tuple[List[float], List[float], List[float], List[float], Normalizer]: A tuple containing the history (values 
            per epoch) for: training loss, validation loss, mean absolute error, R2 score, and `Normalizer` object which
            contains the normalization statistics (mean and std) of the training data (to be used for normalization of
            inputs during inference).
    """
    # Device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss() # Can also consider `SmoothL1Loss` i.e. Huber Loss (middle ground between MSE and L1)
    training_loss_history, validation_loss_history, mae_history, r2_history = [], [], [], []
    
    # Create subdirectory to save models to during training session
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Use timestamp as subdirectory name
    model_type = type(model).__name__
    subdirectory_name = f"{model_type}_{timestamp}"
    save_dir = os.path.join(base_dir, subdirectory_name) # Subdirectory
    os.makedirs(save_dir, exist_ok=True)
    
    # Track model with best R2 score for saving
    current_best_r2 = -1
    
    # Compute normalization statistics for training dataset (used to normalize evaluation inputs as well)
    normalizer = Normalizer()
    normalizer.fit(training_dataset=train_dataloader.dataset) # compute mean and std
    
    # Training step
    for epoch in range(num_epochs):
        epoch += 1 # Account for zero-indexing
        start_time = time.time()
        total_training_loss = 0
        model.train() # Set model to training mode
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Unpack the mini-batch data and move batch data to device
            seq_batch, target_batch = batch
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Normalize inputs
            seq_batch = normalizer(seq_batch)
            
            # Forward pass
            preds = model(seq_batch)
            loss = criterion(preds, target_batch)
            total_training_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the average loss per sample for the epoch to loss_history
        training_loss = total_training_loss / len(train_dataloader.dataset) # Average loss per sample
        training_loss_history.append(training_loss)
        # 2) Evaluate model on validation set
        validation_loss, validation_mae, validation_r2 = evaluate_crypto_model(model, validation_dataloader, normalizer)
        validation_loss_history.append(validation_loss)
        mae_history.append(validation_mae)
        r2_history.append(validation_r2)
        # 3) Record time taken for epoch (training + validation)
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if verbose:
            print(f"Epoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s")
            print(f"(Training) Loss: {training_loss:.4f}")
            print(f"(Validation) Loss: {validation_loss:.4f}, MAE: {validation_mae:.4f}, R2: {validation_r2:.4f}")
                
        model_type: str = type(model).__name__
        # Save model if it has best R2
        if validation_r2 > current_best_r2:
            current_best_r2 = validation_r2 # Update new best R2 score
            model_filename = f"{model_type}_BEST_R2.pth"
            save_model(model, model_filename, save_dir, verbose=verbose)
        
        # Save model after every specified number of epochs
        if epoch % save_interval == 0:
            model_filename = f"{model_type}_epoch{epoch}.pth"
            save_model(model, model_filename, save_dir, verbose=verbose)
            
        if verbose: print("="*90) # Purely visual

    return training_loss_history, validation_loss_history, mae_history, r2_history, normalizer
     
def evaluate_crypto_model(
    model: CryptoBaseModel,
    evaluation_dataloader: DataLoader,
    normalizer: Normalizer,
    device: Optional[torch.device] = None
) -> Tuple[float, float, float]:
    """
    Evaluates the model's performance on the given evaluation dataset. 

    Args:
        model (CryptoBaseModel): Crypto model to evaluate.
        evaluation_dataloader (DataLoader): The dataloader for the evaluation dataset.
        normalizer (Normalizer): Normalizer object that has already been fitted to training data (i.e. normalization
            statistics already computed).
        device: The device to load batch data onto, which should be the same device that the model is on. Defaults to
            None, in which case the device that the model is on will be inferred by checking the model's first parameter.

    Returns:
        Tuple[float, float, float]: Tuple containing the model's average evaluation loss (per sample sequence), 
            mean absolute error, and R2 score. 
    """
    device = device or next(model.parameters()) # Infer the device the model is on by checking the first parameter
    
    model.eval() # Set model to evaluation mode
    criterion = nn.MSELoss()
    total_loss = 0
    mae = torchmetrics.MeanAbsoluteError().to(device)
    r2 = torchmetrics.R2Score().to(device)
    
    with torch.no_grad():  # No gradients needed for evaluation
        for batch in evaluation_dataloader:
            # Unpack the mini-batch data and move batch data to device
            seq_batch, target_batch = batch
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Normalize inputs
            seq_batch = normalizer(seq_batch)
            
            # Forward pass
            preds = model(seq_batch)

            # Update metrics
            loss = criterion(preds, target_batch)
            total_loss += loss.item()
            mae.update(preds, target_batch)
            r2.update(preds, target_batch)

    # Compute final metric values
    final_evaluation_loss = total_loss / len(evaluation_dataloader.dataset) # Average loss per sample
    final_mae = mae.compute().item()
    final_r2 = r2.compute().item()

    return final_evaluation_loss, final_mae, final_r2

def save_model(
    model: nn.Module, 
    model_filename: str, 
    save_dir: str, 
    verbose: bool = True
) -> None:   
    """
    Saves the model's state dictionary (parameters) to the specified directory.

    Args:
        model (nn.Module): The model to save.
        model_filename (str): Name of model parameter file e.g. `mymodel.pth`.
        save_dir (str): Directory to save model's parameters to.
        verbose (bool, optional): Whether to print a confirmation message. Defaults to True.
    """
    model_path = os.path.join(save_dir, model_filename)
    torch.save(model.state_dict(), model_path) # Save state dictionary
    if verbose: print(f"✅ Model saved: {model_path}")

    