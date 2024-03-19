#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
toy example of a non-linearly separable dataset
Hidden Layer(s): At least one hidden layer with a non-linear activation function like ReLU 
added HPO which would do some kind of search in this 3 dimensions:  BS,LR, and number of cells in the hidden layer

by ChatGPT
Result

Best Parameters: {'batch_size': 50, 'learning_rate': 0.1, 'hidden_size': 50}
Best Validation Loss: 0.0199


'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
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
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, criterion, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    # Hyperparameters space
    batch_sizes = [10, 20, 50]
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [5,10, 15, 20,30, 50]
    num_epochs=100
    
    # Generate dataset
    X, Y = generate_dataset(300, 2)

    best_loss = float('inf')
    best_params = {}

    # Grid search
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for hidden_size in hidden_sizes:
                dataset = TensorDataset(X, Y)
                data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                
                model = SimpleNN(2, hidden_size, 1)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Train the model
                train_model(model, criterion, optimizer, data_loader, num_epochs=num_epochs)
                
                # Evaluate the model
                val_loss = evaluate_model(model, criterion, data_loader)
                
                #print(f'BS: {batch_size}, LR: {learning_rate}, Hidden: {hidden_size}, Loss: {val_loss}')
                print('BS: %d, LR: %.4f, Hidden: %d, Loss: %.4f' % (batch_size, learning_rate, hidden_size, val_loss))
                
                # Update best params
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'batch_size': batch_size, 'learning_rate': learning_rate, 'hidden_size': hidden_size}
                    
    print("Best Parameters:", best_params)
    print("Best Validation Loss:", best_loss)

if __name__ == '__main__':
    main()
