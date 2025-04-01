import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_random_directions(model):
    """
    Generate two random direction dictionaries with the same shapes as model parameters.
    """
    d1 = {}
    d2 = {}
    for name, param in model.named_parameters():
        d1[name] = torch.randn_like(param)
        d2[name] = torch.randn_like(param)
    return d1, d2

def normalize_directions(model, d1, d2):
    """
    Normalize each direction so that, for each parameter tensor, the perturbation
    has the same norm as the original parameter.
    
    For convolutional layers (4D tensors), we apply filter normalization (i.e., normalize each output channel independently).
    For fully connected layers (2D tensors), we normalize each row (i.e., each neuron) independently.
    For other parameters, we use global normalization.
    """
    for name, param in model.named_parameters():
        if not (name in d1 and name in d2):
            continue  # Skip parameters without a corresponding random direction

        # Convolutional layers: assume shape (out_channels, in_channels, H, W)
        if param.dim() == 4:
            for i in range(param.size(0)):
                param_norm = torch.norm(param[i])
                d1_norm = torch.norm(d1[name][i])
                d2_norm = torch.norm(d2[name][i])
                if d1_norm > 0:
                    d1[name][i] *= (param_norm / (d1_norm + 1e-10))
                if d2_norm > 0:
                    d2[name][i] *= (param_norm / (d2_norm + 1e-10))
        # Fully connected layers: assume shape (out_features, in_features)
        elif param.dim() == 2:
            for i in range(param.size(0)):
                param_norm = torch.norm(param[i])
                d1_norm = torch.norm(d1[name][i])
                d2_norm = torch.norm(d2[name][i])
                if d1_norm > 0:
                    d1[name][i] *= (param_norm / (d1_norm + 1e-10))
                if d2_norm > 0:
                    d2[name][i] *= (param_norm / (d2_norm + 1e-10))
        # For biases and other parameters (including BatchNorm parameters), use full-tensor normalization
        else:
            param_norm = torch.norm(param)
            d1_norm = torch.norm(d1[name])
            d2_norm = torch.norm(d2[name])
            if d1_norm > 0:
                d1[name] *= (param_norm / (d1_norm + 1e-10))
            if d2_norm > 0:
                d2[name] *= (param_norm / (d2_norm + 1e-10))
    return d1, d2

def compute_loss(model, loss_fn, inputs, targets):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    return loss.item()

def compute_loss_landscape(model, loss_fn, inputs, targets, 
                           d1, d2, 
                           alpha_range=(-1, 1), beta_range=(-1, 1), 
                           resolution=50):
    """
    Compute the loss landscape around the current model parameters.
    Only perturb parameters for which random directions exist (skip buffers like BN.running_mean).
    """
    original_state = copy.deepcopy(model.state_dict())
    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    betas  = np.linspace(beta_range[0], beta_range[1], resolution)
    Loss = np.zeros((resolution, resolution))
    
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            new_state = {}
            for name, param in original_state.items():
                # Only perturb learnable parameters for which we computed directions.
                if name in d1 and name in d2:
                    new_state[name] = param + a * d1[name] + b * d2[name]
                else:
                    new_state[name] = param
            model.load_state_dict(new_state)
            Loss[i, j] = compute_loss(model, loss_fn, inputs, targets)
    
    model.load_state_dict(original_state)
    return alphas, betas, Loss

def plot_loss_landscape(alphas, betas, Loss, use_log_scale=False):
    """
    Plot the loss landscape as a 3D surface and a contour plot.
    """
    A, B = np.meshgrid(betas, alphas)
    if use_log_scale:
        Loss_plot = np.log10(Loss + 1e-15)
        zlabel = 'log10(Loss)'
    else:
        Loss_plot = Loss
        zlabel = 'Loss'
        
    fig = plt.figure(figsize=(14, 6))
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(A, B, Loss_plot, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('Beta direction')
    ax1.set_ylabel('Alpha direction')
    ax1.set_zlabel(zlabel)
    ax1.set_title('3D Loss Landscape')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # Contour Plot
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(A, B, Loss_plot, levels=50, cmap='viridis')
    ax2.set_xlabel('Beta direction')
    ax2.set_ylabel('Alpha direction')
    ax2.set_title('Loss Contours')
    fig.colorbar(contour, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.show()
