# Pytorch Learning Project: MNIST

### Purpose:
The purpose of this project is to learn the fundamentals of machine learning and the Pytorch library.

### Learning Approach:
1. Skim through chosen articles.
2. Reread the articles, but this time more thoroughly.
3. Read through code in "tutorial.py" independant of the articles with the intention of understanding as much as possible.
4. Write my own MNIST program with as little referencing as possible to the tutorial code.
5. Complete the README with an explanation of my code with the intention of utilizing the Feynman Technique.

### Explanation:
The purpose of this explanation is to uncover gaps in my knowledge and hopefully provide those interested with a light explanation of this entry level machine learning project. It should be noted that this is not an attempt to make a completly comprehensive guide.

\[W.I.P\]

    # Define Hyper Params
    learning_rate = 0.01
    num_of_epochs = 10
    batch_size = 64

Hyper perameters are settings made by the programmer on how they want their neural net to behave. By setting `learning_rate = 0.01`, the optimizer (present later in the script) will adjust the paramters (aka the weights) in order to reduce the loss function (also present later in the script) in increments of `0.01`. The value chosen is a standard starting point because of its generally acceptable performance for a veriaty of machine learning tasks. The `num_of_epochs` simply states the amount of iterations the training phase will make over the training data and the `batch_size` states how the data will be broken up which is useful for saving resources.

    # Loading Data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

The first step of creating a machine learning model would be to identify a set of data which patterns could be extracted from. In this case, the MNIST dataset is being used which we can later use to train our model in identifying handwritten numbers from 0-9. Above you can see we defined and loaded 2 different datasets for both training and testing which is not required but a good practice in order to check how versatile the trained model is opposed to checking how well the model memorized answered to the input it was trained on. More specifically, `transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])` defines the original transformation we'll make on the data since the original files containing the data we need are in an unsupported and inefficiant format. `train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)` and `train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)` define and organize the data into a format best for our use case. Lastly, `test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)` and `test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)` do the same thing as when we defined and organized the training data. Except, since we will use this data for training, we specified the need for the training set with `train=False` and turned shuffle off since it will not have an effect on the neural net at that point.

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

After preparing your data, the next step would be creating the neural net you will be training and defining it. A lot of the heavy lifting has already been done which allows us to simply inherit from `torch.nn.Module` when creating our neural net class `Net`. All that is left for us to do is define which layers we want to use which is done within the `Net.__init__` method and use those defined layers in our "forward pass" within the `Net.forward` method. 

Note: a forward pass means inputing data through a neural net and obtaining the result of passing it through the neural net's layers.

the `Net` uses commonplace layers when tackling problems relating to data with spatial features such as MNIST. What follows is a short summary of what each layer does.

- `Conv2d`: This layer is useful for extracting features out of spatial data such as images. In order for it to work, `Conv2d` will require a few hyper perameters such as `in_channels`, `out_channels`, `kernel_size`, and `padding`. The `in_channels` in our case tell the layer how many color channels the input has and since the MNIST dataset is grayscale, this would be set to `1` for the first `Conv2d`. The `out_channels` describe the amount of features you want the layer to try and recognize. These features could be anything but a few examples would be straight lines, curves, or edges which are all seperated into different "feature maps." These feature maps are then outputted to the next layer which is another `Conv2d` in our case, so to account for these feature maps the `in_channels` is set to `32` and to detect further features within those features the `out_channels` are set to `64`. The choice of the value of these hyper perameters seem to be arbitrary and should be tweaked to best fit their particular model. The `padding` simply states how many rows and columns of 0s should be added to an input image so that the "kernels" (aka as filters) can do their operations without error. In short, in a convolutional layer, the kernels are responsible for creating specific feature maps and their number matches the number of `out_channels`. The importance of `kernal_size`, `padding`, and how the kernal creates feature maps should be better explained by this animation:
    ![Convolutional Layer Kernel Gif](assets/2D_Convolution_Animation.gif)

- `Linear`: ...
- `ReLu`: ...
- `MaxPool2d`: ...
- `Dropout`: ...

Note: ...

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

second last thing done...

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

final thing done...