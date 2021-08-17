import numpy

class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
    def feedforward(self, inputs):
        return [i+2 for i in inputs]

        
