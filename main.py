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

train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
train_data_loader = torch.utils.dataloader(train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
test_data_loader = torch.utils.dataloader(test_data, batch_size=batch_size, shuffle=False)

# Creating Model
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernal_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernal_size=5, padding=2)
        self.lin1 = torch.nn.Linear(5*5*64, 1024)
        self.lin2 = torch.nn.Linear(1024, 10)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = torch.nn.Dropout(0.5)
    
    def forward(self, input):
        output = self.conv1(input)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = output.view(-1, 5*5*64)
        output = self.lin1(output)
        output = self.relu(output)
        output = self.drop(output)
        output = self.lin2(output)
        return output

# Training Model