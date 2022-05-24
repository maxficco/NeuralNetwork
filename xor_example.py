from neuralnetwork import * 
import random

def main():
    nn = Network((2,256,16,1))
    print(nn)    
    training_data = [
    [[1,0],[1]],
    [[0,1],[1]],
    [[1,1],[0]],
    [[0,0],[0]]
    ]

    
    for i in range(1000):
        a = random.randint(0,3)
        nn.train(training_data[a][0],training_data[a][1])
    print("Done!")
    
    while True:
        x1 = int(input())
        x2 = int(input())
        asdf = [x1,x2]
        outputs = nn.feedforward(asdf)
        print(outputs[-1])

if __name__=='__main__':
    main()
