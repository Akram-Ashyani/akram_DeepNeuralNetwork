import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler 
from torch.utils.data import DataLoader
import torch.nn.functional as Func
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
testset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Separate 20% of train for validation
indices = list(range(len(trainset)))
np.random.shuffle(indices)
split = int(np.floor(0.2*len(trainset)))

valid_sample = SubsetRandomSampler(indices[:split])
train_sample = SubsetRandomSampler(indices[split:])

# loading train, valid and test
trainloader = DataLoader(trainset, sampler=train_sample, batch_size=64)
validloader = DataLoader(trainset, sampler=valid_sample, batch_size=64)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

fig = plt.figure(figsize= (15,5))
for idx in np.arange(20):
   ax = fig.add_subplot(4, int(20/4), idx+1, xticks=[], yticks=[])
   ax.imshow(np.squeeze(images[idx]), cmap='gray')
   ax.set_title(labels[idx].item())
fig.tight_layout()
plt.show()

class Model(nn.Module):
   def __init__(self):
      super().__init__()
      self.FC1 = nn.Linear(28*28, 512)
      self.FC2 = nn.Linear(512, 256)
      self.FC3 = nn.Linear(256, 128)
      self.FC4 = nn.Linear(128, 10)

      self.Relu = nn.ReLU() # another way to define activation function
      self.dropout = nn.Dropout(0.2)

   def forward(self, x):
      x = x.view(x.shape[0], -1) # change shape to (batch size, 28*28*1)
      x = self.dropout(Func.relu(self.FC1(x)))
      x = self.dropout(self.Relu(self.FC2(x))) # use the defined activation function
      x= self.dropout(Func.relu(self.FC3(x)))
      output = self.FC4(x)
      return output
   
# Another way to define the Model instead of using class like above 
# or it can be used as self.classifier inside of the __init__ then used in forward by output = self.sequential(x)
classifier = nn.Sequential(nn.Linear(28*28, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 256),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(256, 128),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(128, 10))

model = Model()
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 40

# start of training and comparing with evaluation
model.train()
train_losses, valid_losses = [], []
for epoch in range(epochs):
   train_loss = 0
   valid_loss = 0
   for i, (images, labels) in enumerate(trainloader):
      optimizer.zero_grad()
      output = model(images)
      loss = loss_criterion(output, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

   for i, (images, labels) in enumerate(validloader):
      output = model(images)
      loss = loss_criterion(output, labels)
      valid_loss += loss.item()

   train_losses.append(train_loss)
   valid_losses.append(valid_loss)
   print(f'Epoch {epoch+1}/ T_loss {train_loss/len(trainloader.sampler)}/ V_loss {valid_loss/len(validloader.sampler)}')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
total = 0
correct = 0
correct_overall = 0

model.eval() # test step so it will skip dropout
for i, (images, labels) in enumerate(testloader):
   output = model(images)
   loss = loss_criterion(output, labels)

   test_loss = loss.item()
   total += labels.size(0)

   _, pred = torch.max(output, 1)
   correct = np.squeeze(pred.eq(labels.view_as(pred)))
   correct_overall += (pred == labels).sum()
   for i in range(len(labels)):
      label = labels[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1

test_loss = test_loss / len(testloader.sampler)
# print('loss {}'.format(test_loss))
accuracy = 100 * correct_overall / total
print('loss {}, overall accuracy {}'.format(test_loss, accuracy))
for i in range(10):
   if class_total[i] > 0:
      print(f'accuracy of class {i} is: {100*class_correct[i]/class_total[i]}')