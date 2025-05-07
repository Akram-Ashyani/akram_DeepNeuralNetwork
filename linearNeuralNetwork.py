import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt

batch_size = 256
num_epoch = 10
input_dim = 28*28
output_dim = 10
learning_rate = 0.001

train_data = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

class Visualize_data():
    def __init__(self, image, label):
        self.image = image
        self.label = label

    def visualization(self):
        show_image = train_data[self.image][self.label].numpy().reshape(28, 28)
        plt.imshow(show_image, cmap=("gray"))
        plt.title(f"label: {train_data[self.image][1]}")
        plt.show()

visual = Visualize_data(20, 0)
visual.visualization()
