from nn import *

def main():
    nn = Network(2,2,1)
    inputs = [0,1]
    outputs = nn.feedforward(inputs)
    print(outputs)

if __name__=='__main__':
    main()
