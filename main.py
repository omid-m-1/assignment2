import torch
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import importlib
custom_linear = importlib.import_module("deep-codegen.pytorch_apis").custom_linear_with_bias # Import custom linear layer

import numpy as np
import random
import time
import argparse
import os

# Set random seed 
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

# Check available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LeNet model using CUDA function
class LeNet(torch.nn.Module):
    # Initialize weights and biases
    def __init__(self, kernel=True):
        super(LeNet, self).__init__()
        self.kernel = kernel
        if (self.kernel == 'Custom'):
            self.W1 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(300, 28*28, requires_grad=True)))
            self.b1 = torch.nn.Parameter(torch.zeros(300, requires_grad=True))
            self.W2 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(100, 300, requires_grad=True)))
            self.b2 = torch.nn.Parameter(torch.zeros(100, requires_grad=True))
            self.W3 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(10, 100, requires_grad=True)))
            self.b3 = torch.nn.Parameter(torch.zeros(10, requires_grad=True))
        else:
            self.W1 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(28*28, 300, requires_grad=True)))
            self.b1 = torch.nn.Parameter(torch.zeros(300, requires_grad=True))
            self.W2 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(300, 100, requires_grad=True)))
            self.b2 = torch.nn.Parameter(torch.zeros(100, requires_grad=True))
            self.W3 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(100, 10, requires_grad=True)))
            self.b3 = torch.nn.Parameter(torch.zeros(10, requires_grad=True))
    # Define layers
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        if (self.kernel == 'Custom'):
            # Model layers with Custom Cuda linear function
            x = torch.nn.functional.relu(custom_linear(x, self.W1, self.b1)) # First custom layer
            x = torch.nn.functional.relu(custom_linear(x, self.W2, self.b2)) # Second custom layer
            x = torch.nn.functional.relu(custom_linear(x, self.W3, self.b3)) # Third custom layer
        else:
            # Model layers with PyTorch linear layer
            x = torch.nn.functional.relu(x.mm(self.W1) + self.b1) # First PyTorch layer
            x = torch.nn.functional.relu(x.mm(self.W2) + self.b2) # Second PyTorch layer
            x = torch.nn.functional.relu(x.mm(self.W3) + self.b3) # Third PyTorch layer
        return torch.nn.functional.softmax(x, dim=1) # Output layer

# Train function
def train_model(model, num_epochs=5):
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Optimizer

    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device) # Send inputs and labels to device
            optimizer.zero_grad() # reset gradients
            outputs = model(inputs) # Model output
            loss = criterion(outputs, labels) # Compute loss
            loss.backward() # Backward step
            optimizer.step() # update model

            running_loss += loss.item()
            if i % 500 == 499:  # Print every 500 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 500:.4f}')
                running_loss = 0.0

    training_time = time.time() - start_time
    print(f'Training Time: {training_time:.2f} seconds')
    return training_time

# Test function
def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device) # Send inputs and labels to device
            outputs = model(inputs) # Model output
            _, predicted = torch.max(outputs.data, 1) # Predicted class
            total += labels.size(0) # Count test samples
            correct += (predicted == labels).sum().item() # Count correct predictions

    accuracy = 100 * correct / total # Calculate accuracy
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    # Define kernel, batch_size and epochs
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default='Custom', help='valid options: [Custom, PyTorch]')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)

    # Train and evaluate custom model
    print(f"Training {args.kernel} Model...")
    model = LeNet(args.kernel).to(device) # Send inputs and labels to device
    time = train_model(model, args.epochs)
    accuracy = test_model(model)

    # Print results
    print("final Results:")
    print(f"{args.kernel} Model - Time: {time:.2f}s, Accuracy: {accuracy:.2f}%")

