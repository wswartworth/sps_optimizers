import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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

class Medium_Model(nn.Module):
    def __init__(self):
        super(Medium_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.pool(self.relu(self.conv1(x)))
        y = self.pool(self.relu(self.conv2(y)))
        y = y.view(y.shape[0], -1)
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)
        return y

class Large_Model(nn.Module):
    def __init__(self):
        super(Large_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        y = self.pool(self.relu(self.conv1(x)))
        y = self.pool(self.relu(self.conv2(y)))

        y = y.view(y.shape[0], -1)
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)
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


def classification_accuracy(data_loader, model):
  total=0
  correct=0
  with torch.no_grad():
    for images, labels in data_loader:
      images, labels = to_gpu(images), to_gpu(labels)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()

  return correct/total

#Assumes that all batches have same size (fix?)
def total_loss(data_loader, model, loss_fn):
  total = 0
  count = 0
  with torch.no_grad():
    for x,y in data_loader:
      x,y = to_gpu(x), to_gpu(y)
      pred = model(x)
      loss = loss_fn(pred,y)
      total += loss #FIX THIS
      count += 1
  return total/count

###############################################################

cuda_enabled = False #set to True to run on GPU
to_gpu = lambda x: x.cuda() if cuda_enabled else x
if cuda_enabled is False: print("Warning: Not running with GPU")


def run_experiment(model, optimizer, trainloader, testloader, 
                    n_epochs, exp_name, info):

  print("Running experiment " + exp_name)

  criterion = nn.CrossEntropyLoss()


  batch_losses = []
  test_class_accs = []
  train_class_accs = []
  test_losses = []
  train_losses = []


  for epoch in tqdm(range(n_epochs)):

    test_class_accs.append(classification_accuracy(testloader, model))
    train_class_accs.append(classification_accuracy(trainloader, model))
    train_losses.append(total_loss(trainloader, model, criterion))
    test_losses.append(total_loss(testloader, model, criterion))

    for x, y in trainloader:
        optimizer.zero_grad()
        x, y = to_gpu(x), to_gpu(y)
        pred = model(x)
        loss = criterion(pred,y)
        batch_losses.append(loss.item())

        #need create_graph=True for second derivative computations in optimizer
        loss.backward(create_graph=True)

        optimizer.step(loss=loss)

  path = './experiment_data/' + exp_name
  try:
    os.mkdir(path)
  except: #folder already exists
    print("Experiment name already exists. Rewriting...")

  items = [batch_losses, test_class_accs, train_class_accs,
          train_losses, test_losses, model]
  filenames = ["batch_losses", "test_class_accs", "train_class_accs",
            "train_losses", "test_losses", "model"]

  for item, filename in zip(items, filenames):
      full_path = path + "/" + filename + ".pkl"
      print("writing to ", full_path)
      pickle.dump( item, open( full_path, "wb" ) )
    
  with open(path + "/info.txt", "w") as info_file:
      info_file.write(info)


#####################################################################
mnist_trainloader, mnist_testloader = load_MNIST(batch_size=50)
small_model = Small_Model()
med_model = Medium_Model()


testparams = {
"model":small_model, 
"optimizer":sps_optimizers.SGD_test(small_model.parameters()), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":3, 
"exp_name":"test", 
"info": "some test info"}

epochs = 15

SGDparams = {
"model":small_model, 
"optimizer":sps_optimizers.SGD_test(small_model.parameters()), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SGD_small", 
"info": "step size 0.001"}

SP2max_small = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small", 
"info": "lmda=0.5"}

SP2l1_small = {
"model":small_model, 
"optimizer":sps_optimizers.SP2L1_plus(small_model.parameters(), lmda=0.5, init_s=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2l1_small", 
"info": "lmda=0.5, init_s=1"}

SP2l2_small = {
"model":small_model, 
"optimizer":sps_optimizers.SP2L2_plus(small_model.parameters(), lmda=0.5, init_s=1),
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2l2_small", 
"info": "lmda=0.5, init_s=1"}

########################################

SGD_med = {
"model":med_model, 
"optimizer":sps_optimizers.SGD_test(med_model.parameters()), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SGD_med", 
"info": "step size 0.001"}

SP2max_med = {
"model":med_model, 
"optimizer":sps_optimizers.SP2max_plus(med_model.parameters(), lmda=0.5), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_med", 
"info": "lmda=0.5"}

SP2l1_med = {
"model":med_model, 
"optimizer":sps_optimizers.SP2L1_plus(med_model.parameters(), lmda=0.5, init_s=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2l1_med", 
"info": "lmda=0.5, init_s=1"}

SP2l2_med = {
"model":med_model, 
"optimizer":sps_optimizers.SP2L2_plus(med_model.parameters(), lmda=0.5, init_s=1),
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2l2_med", 
"info": "lmda=0.5, init_s=1"}

################## Small stepsize ###########################
SP2max_small_sz_small = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=0.1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_sz_0.1", 
"info": "lmda=0.5, stepsize=0.1"}

SP2max_small_sz_vsmall = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=0.01), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_sz_0.01", 
"info": "lmda=0.5, stepsize=0.01"}

SP2max_small_sz_1 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_sz_1", 
"info": "lmda=0.5, stepsize=1"}

SP2max_small_sz_0 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=0), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":5, 
"exp_name":"SP2max_small_sz_0", 
"info": "lmda=0.5, stepsize=0.00"}

############################# Varying Lambda ##########################
SP2max_small_lmda_9 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.9,
  stepsize=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_lmda_0.9", 
"info": "lmda=0.9, stepsize=1"}

SP2max_small_lmda_5 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_lmda_0.5", 
"info": "lmda=0.5, stepsize=1"}

SP2max_small_lmda_1 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.1,
  stepsize=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_lmda_0.1", 
"info": "lmda=0.1, stepsize=1"}

SP2max_small_lmda_01 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.01,
  stepsize=1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_lmda_0.01", 
"info": "lmda=0.01, stepsize=1"}

################################## momentum ###############################

SP2max_small_beta_9 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1, beta=0.9), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_beta_0.9", 
"info": "lmda=0.5, stepsize=1, beta=0.9"}

SP2max_small_beta_5 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1, beta=0.5), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_beta_0.5", 
"info": "lmda=0.5, stepsize=1, beta=0.5"}

SP2max_small_beta_3 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1, beta=0.3), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_beta_0.3", 
"info": "lmda=0.5, stepsize=1, beta=0.3"}

SP2max_small_beta_0 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1, beta=0.0), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_beta_0.0", 
"info": "lmda=0.5, stepsize=1, beta=0.0"}


SP2max_small_beta_1 = {
"model":small_model, 
"optimizer":sps_optimizers.SP2max_plus(small_model.parameters(), lmda=0.5,
  stepsize=1, beta=0.1), 
"trainloader":mnist_trainloader,
"testloader": mnist_testloader, 
"n_epochs":epochs, 
"exp_name":"SP2max_small_beta_0.1", 
"info": "lmda=0.5, stepsize=1, beta=0.1"}

#run_experiment(**SGDparams)
#run_experiment(**SP2max_small)
#run_experiment(**SP2l1_small)
#run_experiment(**SP2l2_small)

#run_experiment(**SGD_med)
#run_experiment(**SP2max_med)
#run_experiment(**SP2l1_med)
#run_experiment(**SP2l2_med)

#run_experiment(**SP2max_small_sz_1)
#run_experiment(**SP2max_small_sz_small)
#run_experiment(**SP2max_small_sz_vsmall)

#run_experiment(**SP2max_small_lmda_01)
#run_experiment(**SP2max_small_lmda_1)
#run_experiment(**SP2max_small_lmda_5)
#run_experiment(**SP2max_small_lmda_9)

#run_experiment(**SP2max_small_beta_9)
#run_experiment(**SP2max_small_beta_5)
#run_experiment(**SP2max_small_beta_0)
run_experiment(**SP2max_small_beta_3)

#run_experiment(**SP2max_small_beta_1)




