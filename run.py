from nn import *

def main():
    nn = Network((2,3,2))

    inputs = [0,1]
    outputs = nn.feedforward(inputs)
    targets = [1,1]
    errors = nn.calculate_errors(outputs, targets)
    
    #nn.train(inputs,targets)


if __name__=='__main__':
    main()
