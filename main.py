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
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Creating Model
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.lin1 = torch.nn.Linear(7*7*64, 1024)
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
        output = output.view(-1, 7*7*64)
        output = self.lin1(output)
        output = self.relu(output)
        output = self.drop(output)
        output = self.lin2(output)
        return output


model = Net()

# Train Model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_of_epochs):
    for i, (images, labels) in enumerate(train_data_loader):
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Output Relevant Data
        print(f"Epoch: {epoch}/{num_of_epochs}; Step: {i}/{len(train_data_loader)}; Loss: {loss.item():.10f}")

# Test Model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_data_loader:
        output = model(images)

        index, prediction = torch.max(output.data, 1)

        total += labels.size(0)
        correct += (prediction == labels).sum().item()
    
    print(f"Accuracy of Model: {(correct/total)*100}%")
