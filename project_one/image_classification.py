import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# Load and normalize CIFAR10
def transform_data(batch_size: int = 4):
    # Transform the data, normalize and augment
    # Using random crop and horizontal flip for data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return [trainset, trainloader], [testset, testloader], classes
    
def imshow(img):
    img = img / 2 + 0.5     # de-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Custom_Optimizer():
    def __init__(self, params, lr: float = 0.001, momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(param) for param in params]
        
    def zero_grad(self):
        for param in self.params:
            param.grad = torch.zeros_like(param)
            
    def step(self):
        for i, param in enumerate(self.params):
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
            param -= self.velocity[i]

class Net(nn.Module):
    def __init__(self, optimizer = None, criterion = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        if optimizer is None:
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optimizer
            
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, data, device):
        optimizer = self.optimizer
        criterion = self.criterion
        
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss
    
    def evaluate(self, data, device):
        correct = 0
        total = 0
        
        # Not training, dont need to calculate gradients
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                # Forward pass
                outputs = net(inputs)

                # Class with the highest score is the prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total

# main
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Hyper-parameters
    num_epochs = 2
    batch_size = 4
    learning_rate = 0.001
    momentum = 0.9
    
    # Load and normalize CIFAR10
    [trainset, trainloader], [testset, testloader], classes = transform_data(batch_size)
    
    # Define a Convolutional Neural Network
    net = Net()
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    
    print(f'Training the network for {num_epochs} epochs...')
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device), data[1].to(device)

            # # zero the parameter gradients
            # optimizer.zero_grad()

            # # forward + backward + optimize
            # outputs = net(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            
            loss = Net.train(net, data, device)

            # print statistics
            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('\nFinished Training')
    
    print('Saving model...')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print('Model saved')
    
    # Test the network on the test data
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # print images
    imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(PATH, weights_only=True))

    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
    
    # Test the network on the whole dataset
    correct, total = Net.evaluate(net, testloader, device)
            
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    
    # Prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Not training, dont need to calculate gradients
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')