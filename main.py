import torch
from torchvision import datasets, transforms
from models import NN_Graph, NN_Sequential
import random
import numpy as np
import os
import argparse
import time
import resource
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  


# @profile
def train(model, hidden_size, segment_size):
    device = torch.device("cpu")
                          
    if model == 'graph':
        model = NN_Graph(segment_size, hidden_size).to(device)
    else:
        model = NN_Sequential(segment_size, hidden_size).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            data.requires_grad_()

            if segment_size <= 1:
                output = model.forward_without_checkpoint(data)
            
            else:
                output =  model(data)
          

            loss = nn.CrossEntropyLoss()(output, target)
            
            # loss.backward()
            torch.autograd.backward(loss)
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            # break





def main(stats_path, model, segment_size, hidden_size):
    # Run the train() function
    start_time = time.time()
    train(model, hidden_size, segment_size)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    with open(stats_path, 'a+') as f:
        f.write(f"{model},{segment_size},{hidden_size},{peak},{elapsed_time:.2f}\n")


transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=False)

if __name__ == "__main__":


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model and log parameters.')
    parser.add_argument('--stats_path', type=str, help='Path to log status and parameters.')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--segment_size', type=int, help='Segment size for the model.')
    parser.add_argument('--hidden_size', type=int, help='Hidden size parameter.')

    args = parser.parse_args()

    main(args.stats_path, args.model, args.segment_size, args.hidden_size)


