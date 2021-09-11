from nn import * 
import random
import math

def main():
    nn = Network((2,256,16,1))
    
    training_data = [
    [[1,0],[1]],
    [[0,1],[1]],
    [[1,1],[0]],
    [[0,0],[0]]
    ]
    for i in range(5000):
        a = random.randint(0,3)
        nn.train(training_data[a][0],training_data[a][1])
    
    print("\n\n\n\n\n\n")
    while True:
        x1 = int(input())
        x2 = int(input())
        asdf = [x1,x2]
        outputs = nn.feedforward(asdf)
        print(outputs[-1])

if __name__=='__main__':
    main()
