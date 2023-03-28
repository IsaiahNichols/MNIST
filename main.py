# My Recreation of the MNIST Program
import torch
import torchvision

# Steps to complete
# 1) Define Hyper Params
# 2) Load Data
# 3) Create Model
# 4) Train model
# 5) Test model
# 6) Evaluate Performance

# Define Hyper Params
learning_rate = 0.01
num_of_epochs = 10
batch_size = 64

# Loading Data
transform = torchvision.transforms.Compose(torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,)))

train_data = torchvision.datasets.MNIST("./data", train=True, transform = transform, download = True)
train_data_loader = torch.utils.dataloader(train_data, batch_size = batch_size, shuffle = True)

test_data = torchvision.datasets.MNIST("./data", train = False, transform = transform, download = True)
test_data_loader = torch.utils.dataloader(test_data, batch_size = batch_size, shuffle = False)

# Creating Model
