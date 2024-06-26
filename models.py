import torch
import torch.nn as nn
from graph import Graph, Node

class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)

class Block(nn.Module):
    def __init__(self, input_size, hidden_size=64 ):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_size, hidden_size * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x
    
class ConcatBlock(nn.Module):
    def __init__(self, hidden_size, dim=1, n_inputs=2):
        super(ConcatBlock, self).__init__()
        self.concat = Concat(dim)
        self.block = Block(hidden_size * n_inputs, hidden_size)

    def forward(self, x):
        return self.block(self.concat(x))

class LinearBlock(nn.Module):
    def __init__(self, input_shape, hidden_size=64):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_shape, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
       
class NN_Sequential(nn.Module):
    def __init__(self, segment_size=3, hidden_size=16):
        super(NN_Sequential, self).__init__()

        self.graph = Graph(segment_size)
        self.hidden_size = hidden_size

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



class NN_Graph(nn.Module):
    def __init__(self, segment_size=3, hidden_size=16):
        super(NN_Graph, self).__init__()
    
        self.graph = Graph(segment_size)
        self.hidden_size = hidden_size

        # Nodes
        self.graph.add_node(Node('1', nn.Conv2d(1, self.hidden_size, kernel_size=3, stride=1, padding=1)))
        self.graph.add_node(Node("1'", nn.Conv2d(1, self.hidden_size, kernel_size=3, stride=1, padding=1)))

        concat_nodes = {5:3, 9:3}
        
        for i in range(2, 10):
            if i in concat_nodes:
                self.graph.add_node(Node(f"{i}", ConcatBlock(self.hidden_size, n_inputs=concat_nodes[i])))
            else:
                self.graph.add_node(Node(f"{i}", Block(self.hidden_size, hidden_size=self.hidden_size)))
            self.graph.add_node(Node(f"{i}'", Block(self.hidden_size, hidden_size=self.hidden_size)))

        for i in range(10, 15):
            self.graph.add_node(Node(f"{i}", Block(self.hidden_size, hidden_size=self.hidden_size)))  

        self.graph.add_node(Node(str(15), nn.Flatten()))
        self.graph.add_final_node(Node(str(16), nn.Linear(self.hidden_size * 28 * 28, 10)))

        # Edges
        self.graph.add_edge('x', '1')
        self.graph.add_edge('x', "1'")
        
        for i in range(1, 9):
            self.graph.add_edge(f"{i}",  f"{i+1}")
            self.graph.add_edge(f"{i}'", f"{i+1}'")
        
        for i in range(9, 16):
            self.graph.add_edge(f"{i}", f"{i+1}")

        
        # Residuals
        
        self.graph.add_edge(f"{5}'",  f"{5}")
        self.graph.add_edge(f"{3}",  f"{5}")
        

        self.graph.add_edge(f"{9}'",  f"{9}")
        self.graph.add_edge(f"{7}",  f"{9}")


       

            

    def forward(self, x):
        x.requires_grad_()
        return self.graph.forward(x)

    def forward_without_checkpoint(self, x):
        x.requires_grad_()
        
        inputs = {'x' : x}
        for i in range(1, 17):
            node = str(i)
            node_ = str(i) + "'"
            if node_ in self.graph.nodes:
                node_ = self.graph.nodes[str(i) + "'"]
                inputs[node_.name] = node_.module([inputs[input] for input in node_.inputs] if len(node_.inputs) > 1 else inputs[node_.inputs[0]])
            node = self.graph.nodes[str(i)]
            inputs[node.name] = node.module([inputs[input] for input in node.inputs] if len(node.inputs) > 1 else inputs[node.inputs[0]])

        return inputs[self.graph.final]