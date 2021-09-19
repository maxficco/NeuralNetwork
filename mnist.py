from neuralnetwork import * 
import random
import matplotlib.pyplot as plt

def main():
    nn = Network((784,100,10))

    with np.load("mnist.npz") as d:
        training_images = d["training_images"]
        training_labels = d["training_labels"]
        test_images = d["test_images"]
        test_labels = d["test_labels"]

    print("Starting Training...")
    for i in range(100000):
        if i%10000 == 0:
            print(i)
        a = random.randint(0,49999)
        nn.train(training_images[a], training_labels[a])
    print("Done!")
    while True:
        t = int(input())
        plt.imshow(test_images[t].reshape(28,28), cmap="gray")
        plt.show()
        output = nn.feedforward(test_images[t])[-1]
        output = output.round(6).tolist()
        print(output.index(max(output)))
        print(output)
        print(test_labels[t])
if __name__=='__main__':
    main()
