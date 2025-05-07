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

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

class Image_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Image_Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.linear(x)
        return output
    
model = Image_Classifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epoch):
    for i, (image, label) in enumerate(train_loader):
        images = image.view(-1, 28*28)
        labels = label

        #zero_grad
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 200 == 0:
            correct = 0
            total = 0
            for image, label in test_loader:
                images = image.view(-1, 28*28)
                labels = label

                output = model(images)
                _, predict = torch.max(output, 1)
                total += labels.size(0)
                correct += (predict == labels).sum()

            accuracy = 100 * correct / total
            print('Iter {}, Loss {}, Accuracy {}'.format(iter, loss.item(), accuracy))

'''
total_iteration = num_epoch * ceil(len(train_data) //batch_size)= 10 * ceil(60000/256) = 2350 iterations
'''
