# To install everything
# pip install -r requirements.txt

import argparse
import torch
import torch.nn as nn
from utils.lossLandscape import get_random_directions, normalize_directions, compute_loss_landscape, plot_loss_landscape
from utils.dummyModel import FocalLoss, load_mnist_partition

def train_dummy():
    from utils.dummyModel import run_training, load_mnist_partition
    # Train the dummy model (e.g., SimpleCNN) for 10 epochs
    model = run_training(epochs=10, batch_size=64, lr=0.001)
    
    # Load a smaller partition for quick plotting
    X_test, y_test = load_mnist_partition(n_samples=200)
    return model, X_test, y_test

def main():
    
    print("Using dummy model.")
    model, X_test, y_test = train_dummy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    model.to(device)
    model.eval()
    
    # Get two random directions and normalize them
    d1, d2 = get_random_directions(model)
    d1, d2 = normalize_directions(model, d1, d2)
    
    # Compute the loss landscape
    paramRange = 0.1

    alphas, betas, Loss = compute_loss_landscape(
        model,
        nn.CrossEntropyLoss(),
        X_test,
        y_test,
        d1, d2,
        alpha_range=(-paramRange, paramRange),
        beta_range=(-paramRange, paramRange),
        resolution=50
    )
    
    # Plot the loss landscape
    plot_loss_landscape(alphas, betas, Loss, use_log_scale=True)

if __name__ == "__main__":
    main()
