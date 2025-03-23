import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import custom_functions as cf

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
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self, optimizer = None, criterion = None):
        super().__init__()
        self.conv1 = cf.CustomConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = cf.CustomConv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = cf.CustomConv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = cf.CustomConv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = cf.CustomFCLayer(64 * 8 * 8, 256)
        self.fc2   = cf.CustomFCLayer(256, 10)
        
        if optimizer is None:
            # self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            self.optimizer = cf.CustomAdamOptimizer(self.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
            
        if criterion is None:
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = cf.CustomCrossEntropyLoss()
        else:
            self.criterion = criterion

    def forward(self, x):
        # Convolutional layers
        # Block 1
        x = cf.CustomReLU(self.bn1(self.conv1(x)))
        x = cf.CustomReLU(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2
        x = cf.CustomReLU(self.bn3(self.conv3(x)))
        x = cf.CustomReLU(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        x = cf.CustomReLU(self.fc1(x))
        # x = cf.CustomDropout(x, 0.5)
        x = self.fc2(x)

        return cf.CustomSoftmax(x)
    
    def train(self, data, device): 
        optimizer = self.optimizer
        criterion = self.criterion
        
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss

def train_one_epoch(net, trainloader, device, optimizer, criterion):
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward
        outputs = net(inputs)
        
        # Loss
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(trainloader)

def evaluate(net, testloader, device):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    
    return accuracy


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device:\n\t{device}')
    
    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    
    print(f'Hyperparameters:\n\tnum_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, momentum={momentum}')
    
    [trainset, trainloader], [testset, testloader], classes = transform_data(batch_size)
    
    # Create net & send to device
    net = Net().to(device)

    # Choose custom or PyTorch optimizer/loss
    custom_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)
    custom_criterion = cf.CustomCrossEntropyLoss()
    
    print(f'Optimizer:\n\t{custom_opt.__class__.__name__}')
    print(f'Criterion:\n\t{custom_criterion.__class__.__name__}')
    
    print(f'Beginning training for {num_epochs} epochs...')
    
    training_loss_values = []
    training_acc_values = []
    testing_acc_values = []

    for epoch in range(num_epochs):
        new_lr = cf.CustomCosineRateDecay(learning_rate, epoch, num_epochs)
        custom_opt.lr = new_lr

        # Training
        epoch_loss = train_one_epoch(net, trainloader, device, custom_opt, custom_criterion)
        
        # Evaluation
        with torch.no_grad():
            train_acc = evaluate(net, trainloader, device)
            test_acc = evaluate(net, testloader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss:.3f}, Test Acc={test_acc:.2f}%, Train Acc={train_acc:.2f}%")

        # Save the loss and accuracy for plotting
        training_loss_values.append(epoch_loss)
        training_acc_values.append(train_acc)
        testing_acc_values.append(test_acc)



    print('Finished Training')

    # Plot
    fig, axs = plt.subplots(1,3)
    axs[0].plot(training_loss_values)
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[1].plot(training_acc_values)
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[2].plot(testing_acc_values)
    axs[2].set_title('Test Accuracy')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Accuracy (%)')
    plt.show()
    
    # Save
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print("Model saved.")

    # Load
    net_with_weights = Net().to(device)
    net_with_weights.load_state_dict(torch.load(PATH, weights_only=True))
    acc = evaluate(net_with_weights, testloader, device)
    print(f"Accuracy after loading: {acc:.2f}%")