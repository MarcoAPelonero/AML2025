import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
import numpy as np
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for MNIST classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))  
        x = self.pool(x)              
        x = self.relu(self.conv2(x))  
        x = self.pool(x)              
        x = x.view(x.size(0), -1)     
        x = self.relu(self.fc1(x))    
        x = self.fc2(x)               
        return x

def load_mnist_partition(n_samples=5000):
    """
    Loads a partition of the MNIST dataset using sklearn's fetch_openml.
    Returns the first n_samples examples.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist['data']
    y = mnist['target'].astype(np.int64)
    
    X = X / 255.0
    X = X.reshape(-1, 1, 28, 28)
    
    X_part = X[:n_samples]
    y_part = y[:n_samples]
    return X_part, y_part

def run_training(epochs=10, batch_size=64, lr=0.001):
    """
    Creates and trains the CNN on a partition of MNIST for a few epochs.
    Uses CUDA if available. Returns the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    X, y = load_mnist_partition(n_samples=5000)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # probability for correct class
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            # alpha should be a tensor of shape [num_classes]
            at = self.alpha[targets]
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss