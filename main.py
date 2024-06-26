import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from memory_profiler import profile, memory_usage
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential, set_checkpoint_early_stop
import tracemalloc
from models import NN
import random
import numpy as np
import os
import torch
import resource


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
def train():
    device = torch.device("cpu")
                          
                   
    model = NN().to(device)

    

    # nodes = {}
    # for node in model.graph.nodes:
    #     module = model.graph.nodes[node].module
    #     if  isinstance(module, nn.Linear):
    #         nodes[node] = module.weight.clone()

    print(len(list(model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            data.requires_grad_()



            # output =  model(data)
            # # output = checkpoint_sequential(model, 3, data, use_reentrant=False)

            output =  model.forward_without_checkpoint(data)
          

            loss = nn.CrossEntropyLoss()(output, target)
            
            # loss.backward()
            torch.autograd.backward(loss)
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            break

    # for node in model.graph.nodes:
    #     module = model.graph.nodes[node].module
    #     if  isinstance(module, nn.Linear):
    #         print(node, torch.all(nodes[node] == module.weight) )

# Preparando os dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=False)



train()

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
