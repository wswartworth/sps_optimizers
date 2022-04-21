import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#import pickle
#import os
from tqdm import tqdm

import sps_optimizers

class Small_Model(nn.Module):
    def __init__(self):
        super(Small_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        y = self.pool(self.relu(self.conv1(x)))
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        return y

def load_MNIST(batch_size=50):
  transform_train = transforms.Compose([
    transforms.ToTensor()
    ,transforms.Normalize((0.1307,), (0.3081,))
  ])
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  train_data = torchvision.datasets.MNIST(root="./", train=True,download=True, transform=transform_train)
  test_data = torchvision.datasets.MNIST(root="./", train=False,download=True, transform=transform_test)


  trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

  return trainloader, testloader


###############################################################

cuda_enabled = False #set to True to run on GPU
to_gpu = lambda x: x.cuda() if cuda_enabled else x
if cuda_enabled is False: print("Warning: Not running with GPU")


trainloader, testloader = load_MNIST(batch_size=50)
criterion = nn.CrossEntropyLoss()

model = Small_Model()

n_epochs = 1

batch_losses = []

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
#optimizer = sps_optimizers.SGD_test(model.parameters())
#optimizer = sps_optimizers.SP2max_plus(model.parameters(), lmda=0.5)
#optimizer = sps_optimizers.SP2L1_plus(model.parameters(), lmda=0.5, init_s=1)
optimizer = sps_optimizers.SP2L2_plus(model.parameters(), lmda=0.5, init_s=1)

for epoch in tqdm(range(n_epochs)):
      for x, y in trainloader:
            optimizer.zero_grad()
            x, y = to_gpu(x), to_gpu(y)
            pred = model(x)
            loss = criterion(pred,y)
            batch_losses.append(loss.item())

            #need create_graph=True for second derivative computations in optimizer
            loss.backward(create_graph=True)

            optimizer.step(loss=loss)

plt.plot(batch_losses)
plt.show()






