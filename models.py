import torch
import torch.nn as nn
from graph import Graph, Node

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.graph = Graph()
        self.hidden_size = 32

        self.graph.add_node(Node('1', nn.Conv2d(1, self.hidden_size, kernel_size=3, stride=1, padding=1)))

        for i in range(2, 7):
            self.graph.add_node(Node(str(i), nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)))

        final_layers = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.hidden_size, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128*7*7, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        ]

        for i, layer in enumerate(final_layers):
            self.graph.add_node(Node(str(i+7), layer))

        self.graph.add_final_node(Node(str(len(final_layers)+7), nn.Linear(self.hidden_size, 10)))

        self.graph.add_edge('x', '1')
        
        for i in range(1, len(self.graph.nodes)):
            self.graph.add_edge(str(i), str(i+1))

    def forward(self, x):
        x.requires_grad_()
        return self.graph.forward(x)

    def forward_without_checkpoint(self, x):
        x.requires_grad_()
        for i in range(1, len(self.graph.nodes) + 1):
            x = self.graph.nodes[str(i)].module(x)
        return x