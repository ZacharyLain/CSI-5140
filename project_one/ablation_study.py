import torch

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

import custom_image_classification as cic
import custom_functions as cf

def print_header(function_name: str) -> None:
    print(f'{"="*49}')
    print(f'=\t\tAbalation Study\t\t\t=')
    print(f'=\t\t{function_name}\t\t=')
    print(f'{"="*49}')

def plot(function_name: str, training_loss_values: list, training_acc_values: list, testing_acc_values: list) -> None:
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
    
    # Ensure the directory exists before saving the file
    os.makedirs('./plots', exist_ok=True)
    
    # Save the plot to ./plots/ablation_<function_name>.png
    plt.savefig(f'./plots/ablation_{function_name}.png')
    plt.close(fig)


def ablation_study(optimizer, criterion, lr_decay_function) -> Union[list, list, list]:
    custom_opt = optimizer
    custom_criterion = criterion

    print(f'Optimizer:\n\t{custom_opt.__class__.__name__}')
    print(f'Criterion:\n\t{custom_criterion.__class__.__name__}')
    
    print(f'Beginning training for {num_epochs} epochs...')
    
    training_loss_values = []
    training_acc_values = []
    testing_acc_values = []

    for epoch in range(num_epochs):
        # Default to Cosine Rate Decay
        if lr_decay_function == cf.CustomStepRateDecay:
            new_lr = cf.CustomStepRateDecay(learning_rate, 0.1)
            custom_opt.lr = new_lr
        else:
            new_lr = cf.CustomCosineRateDecay(learning_rate, epoch, num_epochs)
            custom_opt.lr = new_lr

        # Training
        epoch_loss = cic.train_one_epoch(net, trainloader, device, custom_opt, custom_criterion)
        
        # Evaluation
        with torch.no_grad():
            train_acc = cic.evaluate(net, trainloader, device)
            test_acc = cic.evaluate(net, testloader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss:.3f}, Test Acc={test_acc:.2f}%, Train Acc={train_acc:.2f}%")

        # Save the loss and accuracy for plotting
        training_loss_values.append(epoch_loss)
        training_acc_values.append(train_acc)
        testing_acc_values.append(test_acc)

    print('Finished Training')
    
    # Save
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print("Model saved.")

    # Load
    net_with_weights = cic.Net().to(device)
    net_with_weights.load_state_dict(torch.load(PATH, weights_only=True))
    acc = cic.evaluate(net_with_weights, testloader, device)
    print(f"Accuracy after loading: {acc:.2f}%")

    return training_loss_values, training_acc_values, testing_acc_values

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device:\n\t{device}')
    
    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    momentum = 0.9
    
    print(f'Hyperparameters:\n\tnum_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, momentum={momentum}')
    
    [trainset, trainloader], [testset, testloader], classes = cic.transform_data(batch_size)
    
    # Create net & send to device
    net = cic.Net().to(device)
    
    # Create a list of functions so that the ablation study can be automated
    learning_rate_decay_functions = [cf.CustomCosineRateDecay, cf.CustomStepRateDecay]
    regularization_functions = [cf.CustomL2Regularization, cf.CustomDropout]
    lambda_values = [0.001, 0.01, 0.1]
    optimizers = [cf.CustomSGDMomentumOptimizer, cf.CustomAdamOptimizer]
    beta1_values = [0.9, 0.99]
    beta2_values = [0.99, 0.999]

    # Base optimzer and loss function (pytorch called this criterion)
    base_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)
    base_criterion = cf.CustomCrossEntropyLoss()

    # # Loop through all the functions to see how they affect the model
    # for learning_rate_decay_function in learning_rate_decay_functions:
    #     # Reinitialize the model
    #     net = cic.Net().to(device)

    #     # Reinitialize the optimizer with the new model parameters
    #     base_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)

    #     print_header(learning_rate_decay_function.__name__)
    #     train_loss, test_acc, test_acc = ablation_study(base_opt, base_criterion, learning_rate_decay_function)

    #     plot(learning_rate_decay_function.__name__, train_loss, test_acc, test_acc)

    for regularization_function in regularization_functions:
        # Only L2 regularization has a lambda value
        if regularization_function == cf.CustomL2Regularization:
            for lambda_value in lambda_values:
                # Reinitialize the model
                net = cic.Net().to(device)

                # Reinitialize the optimizer with the new model parameters
                base_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)

                print_header(regularization_function.__name__)
                train_loss, test_acc, test_acc = ablation_study(base_opt, base_criterion, cf.CustomCosineRateDecay)

                plot(f'{regularization_function.__name__}_lambda_{lambda_value}', train_loss, test_acc, test_acc)
        else:
            # Reinitialize the model
            net = cic.Net().to(device)

            # Reinitialize the optimizer with the new model parameters
            base_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)

            print_header(regularization_function.__name__)
            train_loss, test_acc, test_acc = ablation_study(base_opt, base_criterion, cf.CustomCosineRateDecay)

    # for optimizer in optimizers:
    #     # Reinitialize the model
    #     net = cic.Net().to(device)

    #     # Optimizer gets reinitialized in the ablation study function below
    #     print_header(optimizer.__name__)
    #     train_loss, test_acc, test_acc = ablation_study(optimizer(list(net.parameters()), lr=learning_rate), base_criterion, cf.CustomCosineRateDecay)

    #     plot(optimizer.__name__, train_loss, test_acc, test_acc)

    # for beta1_value in beta1_values:
    #     # Reinitialize the model
    #     net = cic.Net().to(device)

    #     # Reinitialize the optimizer with the new model parameters
    #     base_opt = cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate)

    #     for beta2_value in beta2_values:
    #         print_header(f'Beta1={beta1_value}, Beta2={beta2_value}')
    #         train_loss, test_acc, test_acc = ablation_study(cf.CustomAdamOptimizer(list(net.parameters()), lr=learning_rate, beta1=beta1_value, beta2=beta2_value), base_criterion, cf.CustomCosineRateDecay)

    #         plot(f'Beta1_{beta1_value}_Beta2_{beta2_value}', train_loss, test_acc, test_acc)
    
