import numpy as np

class Network:
	def __init__(self, layer_sizes, activation_func="sigmoid"):
		self.layer_sizes = layer_sizes
		self.activation_func = activation_func
		self.learning_rate = 0.1
		self.weights = []
		self.biases = []
		for i in range(len(self.layer_sizes)):
			if i != 0:
				self.weights.append(np.random.standard_normal((self.layer_sizes[i], self.layer_sizes[i-1])))
				self.biases.append(np.zeros((self.layer_sizes[i],1)))
	def feedforward(self, inputs):
		print("\nWeights per layer:\n")
		for w in self.weights:
			print(w, "\n")
			print("\nBiases per layer:\n")
		for b in self.biases:
			print(b, "\n")
		inputs = np.array(inputs).reshape((self.layer_sizes[0],1))
		print("\nInputs: \n"+ str(inputs))
		hidden_outputs = []
		for w,b in zip(self.weights,self.biases):
			outputs = np.matmul(w, inputs) + b
			outputs = self.activation(outputs)
			hidden_outputs.append(outputs)
			inputs = outputs
		hidden_outputs.pop()
		print("\nOutputs in Hidden Layers:\n" + str(hidden_outputs))
		print("\nOutput of Neural Network:\n" + str(outputs))
		return outputs, hidden_outputs
	
	def calculate_errors(self, outputs, targets):
		targets = np.array(targets).reshape((self.layer_sizes[-1],1))
		errors = []
		errors.append(targets - outputs)

		weights_T = list(map(np.transpose, self.weights))
		weights_T.reverse() # weights_t was going left->right, errors is going right->left	
		weights_T.pop() # first matrix of weights don't matter (we aren't calculating error of inputs)
		
		for count, wt in enumerate(weights_T):
			errors.append(np.matmul(wt, errors[count]))

		print("\nErrors of Outputs and Hidden Layers:\n")
		for err in errors:
			print(err, "\n")

		return errors


	def train(self, inputs, targets):
		outputs, hidden_outputs = self.feedforward(inputs)
		errors = self.calculate_errors(outputs, targets)
		
		gradients = outputs * (1-outputs)
		gradients *= errors[0]
		gradients *= self.learning_rate
		print("\n\n\n"+str(gradients))

	def activation(self, x):
		if self.activation_func == "sigmoid":
			return 1/(1+np.exp(-x)) 
		elif self.activation_func == "relu":
			return np.maximum(x,0)
		
