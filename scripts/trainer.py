from typing import Callable

import torch
import torch_geometric

from torch.optim.lr_scheduler import ExponentialLR

from scripts.classifier import GAT
from scripts.metrics import rmse_metric, accuracy_metric, f1_score_metric


def train_model(
    model: torch.nn.Module, 
    loss_fn: Callable,
    optimizer = None, 
    scheduler = None,
    train_loader = None, 
    valid_loader = None, 
    infer_loader = None, 
    num_epochs: int = 100, 
    logger_freq: int = 10,
    verbose: bool = True,
) -> tuple[
    torch.nn.Module, 
    list[float], 
    list[float], 
    list[float], 
    list[float], 
    float, 
    float
]:

    if verbose:
        valid = "EXCLUDED" if valid_loader is None else "INCLUDED"
        print(f"Starting training... for {num_epochs} epochs\t Validation = {valid}\n")
    
    train_losses, train_metric, valid_losses, valid_metric = [[] for _ in range(4)]
    
    for epoch in range(num_epochs):

        # Train:
        train_loss, train_met = epoch_step(train_loader, model, loss_fn, optimizer)
        train_losses.append(train_loss)
        train_metric.append(train_met)
        
        # Valid:
        valid_loss, valid_met = epoch_step(valid_loader, model, loss_fn, optimizer=None)
        valid_losses.append(valid_loss)
        valid_metric.append(valid_met)
            
        lr = optimizer.param_groups[-1]['lr']
        scheduler.step()
        
        # Print epoch update:
        logger_freq = num_epochs // 10 if logger_freq is None else logger_freq
        if verbose:
            if epoch % logger_freq == 0:
                print("Epoch {} | Learning rate: {:0.6f} | "
                      "Train Loss {:0.4f} | Train Metric {:0.2f} | "
                      "Valid Loss {:0.4f} | Valid Metric {:0.2f} |"
                      "".format(epoch, lr, train_loss, train_met, valid_loss, valid_met))
                
    if verbose:
        print(f"\nFinished training...")
    
    # Test:
    test_loss, test_met = epoch_step(infer_loader, model, loss_fn, optimizer=None)
    if verbose:
        print(f"\nRunning inference on test data...\t"
           "Test Loss {:0.4f} | Test Metric {:0.4f}\n\n".format(test_loss, test_met))
          
    return model, train_losses, train_metric, valid_losses, valid_metric, test_loss, test_met


def build_model(
    num_features: int, 
    embedding_size: int, 
    num_heads: int, 
    dropout_prob: float, 
    use_batch_norm: bool,
    learning_rate: float,
    weight_decay: float,
    scheduler_gamma: float,
) -> tuple[
    torch.nn.Module, 
    torch.optim.Adam, 
    torch.optim.lr_scheduler.ExponentialLR
]:
    """Build the model object"""
    
    # Construct the model
    model = GAT(num_features, embedding_size, num_heads, dropout_prob, use_batch_norm)
    print("GAT model architecture: ", model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # Define the optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay, 
    )

    scheduler = ExponentialLR(
        optimizer, 
        gamma=scheduler_gamma,
    )
      
    return model, optimizer, scheduler


def epoch_step(data_loader: torch_geometric.loader.dataloader.DataLoader, 
               model: torch.nn.Module, 
               loss_fn: Callable,
               optimizer: torch.optim.Adam = None, 
               problem_type: str = "classification"
              ) -> tuple[float, float]:

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    losses, metric = [], []
    
    for batch in data_loader:
      
        # Reset gradients
        if optimizer is not None:
            optimizer.zero_grad() 
        
        # Passing the node features and the connection info
        prediction, _ = model(batch.x, batch.edge_index, batch.batch) 
        true = batch.y
        
        # Calculating the loss and update the gradients
        loss = loss_fn(prediction, true)  

        if optimizer is not None:
            loss.backward()  
            optimizer.step()   
        
        # Calculate metric
        if problem_type == "regression":
            m = rmse_metric(prediction, batch.y)
            metric.append(m.item())
            
        if problem_type == "classification":
            arg = prediction.argmax(dim=1)
            m = accuracy_metric(arg, true)
            
        metric.append(m.item())
        losses.append(loss.item())
        
    return torch.mean(torch.tensor(losses)), torch.mean(torch.tensor(metric))
