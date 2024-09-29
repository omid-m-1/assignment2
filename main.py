import torch
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import importlib
custom_linear = importlib.import_module("deep-codegen.pytorch_apis").custom_linear_with_bias

import numpy as np
import random
import time
import os

#
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
class LeNet(torch.nn.Module):
    #
    def __init__(self):
        super(LeNet, self).__init__()
        self.W1 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(300, 28*28, requires_grad=True)))
        self.b1 = torch.nn.Parameter(torch.zeros(300, requires_grad=True))
        self.W2 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(100, 300, requires_grad=True)))
        self.b2 = torch.nn.Parameter(torch.zeros(100, requires_grad=True))
        self.W3 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(10, 100, requires_grad=True)))
        self.b3 = torch.nn.Parameter(torch.zeros(10, requires_grad=True))
    #
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = torch.nn.functional.relu(custom_linear(x, self.W1, self.b1))
        x = torch.nn.functional.relu(custom_linear(x, self.W2, self.b2))
        x = torch.nn.functional.relu(custom_linear(x, self.W3, self.b3))
        return torch.nn.functional.softmax(x, dim=1)


class LeNet2(torch.nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.W1 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(28*28, 300, requires_grad=True)))
        self.b1 = torch.nn.Parameter(torch.zeros(300, requires_grad=True))
        self.W2 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(300, 100, requires_grad=True)))
        self.b2 = torch.nn.Parameter(torch.zeros(100, requires_grad=True))
        self.W3 = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(100, 10, requires_grad=True)))
        self.b3 = torch.nn.Parameter(torch.zeros(10, requires_grad=True))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        
        # Model layers with PyTorch linear layer
        x = torch.nn.functional.relu(x.mm(self.W1) + self.b1)
        x = torch.nn.functional.relu(x.mm(self.W2) + self.b2)
        x = torch.nn.functional.relu(x.mm(self.W3) + self.b3)
        return torch.nn.functional.softmax(x, dim=1)

#
def train_model(model, num_epochs=1):
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    training_time = time.time() - start_time
    print(f'Training Time: {training_time:.2f} seconds')
    return training_time

def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
    return accuracy


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=1, pin_memory=False)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=1, pin_memory=False)

# Train and evaluate custom model
print("Training Custom Model...")
model = LeNet().to(device)
custom_time = train_model(model)
custom_accuracy = test_model(model)

# Train and evaluate PyTorch model
print("\nTraining PyTorch Model...")
torch_model = LeNet2().to(device)
torch_time = train_model(torch_model)
torch_accuracy = test_model(torch_model)

# Compare results
print("\nComparison Results:")
print(f"Custom Model - Time: {custom_time:.2f}s, Accuracy: {custom_accuracy:.2f}%")
print(f"PyTorch Model - Time: {torch_time:.2f}s, Accuracy: {torch_accuracy:.2f}%")

