import torch
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint
from torchviz import make_dot
from torch.utils.checkpoint import checkpoint_sequential, checkpoint, set_checkpoint_early_stop
from collections import defaultdict
from math import ceil

class Node(nn.Module):
    def __init__(self, name, module):
        super(Node, self).__init__()
        self.name = name
        self.module = module
        self.inputs = []

    def forward(self, x):
        # print(f"Node {self.name} input shape: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        output = self.module(x)

        return output


class Graph(nn.Module):
    def __init__(self, segment_size=3):
        super(Graph, self, ).__init__()

        self.nodes = nn.ModuleDict()
        self.edges = defaultdict(list)
        self.final = None
        self.segment_size = segment_size
        self.dp = {}
        self.checkpoints = {}

    def add_node(self, node):
        self.nodes[node.name] = node

    def add_final_node(self, node):
        self.add_node(node)
        self.final = node.name

    def add_edge(self, from_node, to_node):
        to_node = self.nodes[to_node]
        to_node.inputs.append(from_node)
        self.edges[from_node].append(to_node)


    def run_with_checkpoints(self, functions, segments, input):
       
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                
                return input

            return forward

        segments = min(segments, len(functions))
        segment_size = len(functions) // segments

      
        end = -1


        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            
            input.requires_grad_()
            input = checkpoint(
                run_function(start, end, functions),
                input,
                use_reentrant=False,
            )



        
        return run_function(end + 1, len(functions) - 1, functions)(input)

    def forward(self, input):

        self.nodes['x'] = Node('x', lambda x: x)

        def dfs(node):
            if node.name in self.dp:
                return self.dp[node.name]

            # caso onde não tem inputs
            if not node.inputs:
                return []

            # caso onde tem apenas um input
            if len(node.inputs) == 1:
                prev = self.nodes[node.inputs[0]]
                functions = dfs(prev)
                
                self.dp[node.name] = functions + [lambda x:  node(x)]
                return self.dp[node.name]

            # caso onde tem mais de um input
            else:
                childrens = {}
                for prev in node.inputs:
                    prev = self.nodes[prev]
                    functions = dfs(prev)
                    childrens[prev.name] = functions

                def _func(x):
                    inputs = {}
                    for key in node.inputs:
                        if len(childrens[key]) > 1:
                            segments = ceil(len(childrens[key]) /  self.segment_size)
                            out = self.run_with_checkpoints(childrens[key], segments, x)

                        else:
                            out = childrens[key][0](x)
                        inputs[key] = out
                    
                    inputs = [inputs[key] for key in node.inputs]
                    return  node(inputs)

                self.dp[node.name] = [_func]
                return self.dp[node.name]

        functions = dfs(self.nodes[self.final])
        segments = ceil(len(functions) / self.segment_size )
        
        out = self.run_with_checkpoints(functions, segments, input)
        return out
   