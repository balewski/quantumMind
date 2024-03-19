#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
toy example of a non-linearly separable dataset
Hidden Layer(s): At least one hidden layer with a non-linear activation function like ReLU 

SimpleNN(
  (layer1): Linear(in_features=2, out_features=5, bias=True)
  (relu): ReLU()
  (layer2): Linear(in_features=5, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                    [-1, 5]              15
              ReLU-2                    [-1, 5]               0
            Linear-3                    [-1, 1]               6
           Sigmoid-4                    [-1, 1]               0
================================================================
Total params: 21

Result:
Epoch [91/100], Loss: 0.2576, Accuracy: 96.00%

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchsummary import summary

# Define the network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        #self.relu = nn.Sigmoid()  # all other work as well
        #self.relu = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

def generate_dataset(n_samples, n_features):
    X = np.random.uniform(-1, 1, size=(n_samples, n_features))
    Y = np.where(np.linalg.norm(X, axis=1) < 0.8, 1, 0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).view(-1, 1)

def train_model(model, criterion, optimizer, data_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.round()  # Threshold at 0.5 for binary classification
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        if epoch%5==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def main():
    input_size = 2
    hidden_size = 6 #  6+ gives ~95% accuracy
    output_size = 1
    num_epochs = 100
    batch_size = 10
    learning_rate = 0.1

    X, Y = generate_dataset(300, input_size)
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(model)
    # Summary of the model
    summary(model, input_size=(input_size,))

    train_model(model, criterion, optimizer, data_loader, num_epochs)

if __name__ == '__main__':
    main()
