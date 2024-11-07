# Importing dependencies

import torch
import torchvision
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR


def draw_weights(synapses, Kx, Ky):
    fig = plt.figure()
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
            yy += 1
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    fig.canvas.draw()
    
def draw_weights3d(synapses, Kx, Ky, ax=None):
    # synapses: the weights
    fig = plt.figure()
    Kw = int(np.sqrt(synapses.shape[1]//3)) # i.e. 32
    yy=0
    HM=np.zeros((Kw*Ky, Kw*Kx, 3))
    for y in range(Ky):
        for x in range(Kx):
            HM[y*Kw:(y+1)*Kw,x*Kw:(x+1)*Kw]=synapses[yy,:Kw*Kw*3].reshape(Kw, Kw, 3)
            yy += 1
   
    nc=np.amax(np.absolute(HM))
    tmp = (HM-HM.min())
    tmp /= tmp.max() 
    tmp *= 255
    tmp = tmp.astype(np.uint8)
    if ax is not None:
        im = ax.imshow(tmp)
        ax.axis('off')
    else:
        plt.clf()
        im=plt.imshow(tmp.astype(np.uint8))
        plt.axis('off')
    fig.canvas.draw() 
    
def unsupervised_bio_learning(train_dataset, n_hidden=28, n_epochs=500, batch_size=100,
                              learning_rate=0.01, precision=0.1, 
                              anti_hebbian_learning_strength=0.3, 
                              lebesgue_norm=2, rank=5, skip = 1):
    """
    Unsupervised bio learning function.

    Parameters:
    - train_dataset: Input dataset (torch.Tensor), where each row is a training example.
    - n_input: number of input neurons
    - n_hidden: Number of hidden units (neurons).
    - n_epochs: Number of epochs to train the model.
    - batch_size: The size of the minibatch used to update the weights.
    - learning_rate: Initial learning rate that decreases over epochs.
    - precision: A threshold to normalize the gradient to avoid very small updates.
    - Δ: anti_hebbian_learning_strength: Strength of anti-Hebbian learning (penalizing neurons with low activation).
    - p: lebesgue_norm: Parameter for the Lebesgue norm used to weigh the contributions of the weights.
    - k: rank: Number of hidden neurons that are penalized using anti-Hebbian learning.
    - skip: print the number of epochs every skip-times
    """
    
    # compute n_input, meaning the lenght of the flattened image 
    input_data = torch.stack([data[0].flatten() for data in train_dataset]) 
    n_input = input_data.shape[1] 
    
    # Initialize the synapse weights randomly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synapses = torch.rand((n_hidden, n_input), dtype=torch.float).to(device)
    
    # Loop over epochs
    for epoch in range(n_epochs):
        if (epoch % skip == 0): 
            print('epoch -->', epoch)

        eps = learning_rate * (1 - epoch / n_epochs)

        # Shuffle dataset
        input_data = torch.stack([data[0].flatten() for data in train_dataset]) 
        shuffled_epoch_data = input_data[torch.randperm(input_data.shape[0]),:]

        # Loop through minibatches of input data
        for i in range(len(train_dataset) // batch_size):
            mini_batch = shuffled_epoch_data[i * batch_size:(i + 1) * batch_size, :].to(device)
            mini_batch = torch.transpose(mini_batch, 0, 1)

            # ---Currents---#
            sign = torch.sign(synapses)
            W = sign * torch.abs(synapses) ** (lebesgue_norm - 1)
            currents = torch.mm(W, mini_batch)

            # ---- Activation ----#
            y = torch.argsort(currents, dim=0)

            # Initialize the Hebbian and anti-Hebbian activations matrix
            yl = torch.zeros((n_hidden, batch_size), dtype=torch.float).to(device)

            # Apply Hebbian learning to the neuron with the highest activation
            yl[y[n_hidden - 1, :], torch.arange(batch_size)] = 1.0

            # Apply anti-Hebbian learning to a number of neurons with lower activation
            yl[y[n_hidden - rank], torch.arange(batch_size)] = - anti_hebbian_learning_strength

            # Compute the contribution of the activations on the total input received
            xx = torch.sum(yl * currents, 1)

            # Expand xx to have the same dimensions as the weight matrix
            xx = xx.unsqueeze(1)
            xx = xx.repeat(1, n_input)

            #---Compute change of weights---#
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * synapses

            # Normalize the gradient to prevent very large or very small updates
            nc = torch.max(torch.abs(ds))
            if nc < precision:
                nc = precision

            # Update the synapse weights 
            synapses += eps * (ds / nc)

    return synapses 

def save_state_weights(synapses, data_name, n_hidden, anti_hebbian_learning_strength=0.3, 
                       lebesgue_norm=2, rank=5, Kx=10, Ky=10):
    
    # Dizionario dei nomi dei dataset
    data_name_dict = {1: "MNIST", 2: "CIFAR10", 3: "FashionMNIST"}
    dataset_name = data_name_dict.get(data_name, "UnknownDataset")

    # Costruzione del nome del file in base alle variabili
    file_name = f"output/{dataset_name}_biolinear_hidden{n_hidden}_ahls{anti_hebbian_learning_strength}_lnorm{lebesgue_norm}_rank{rank}"
    
    # Salvataggio dello stato di torch
    torch_file = f"{file_name}.pt"
    torch.save(synapses, torch_file)
    print(f"File torch salvato come: {torch_file}")
    
    # Salvataggio della figura
    if data_name == 2:
        draw_weights3d(synapses, Kx=Kx, Ky=Ky)
    else:
        draw_weights(synapses, Kx=Kx, Ky=Ky)
    fig_file = f"{file_name}.png"
    plt.savefig(fig_file)
    print(f"Figura salvata come: {fig_file}")

class BioClassifier(nn.Module):
    def __init__(self, W_unsupervised, out_features):
        super(BioClassifier, self).__init__()
        
        # Set unsupervised weights and freeze them
        self.W_unsupervised = nn.Parameter(W_unsupervised.T, requires_grad=False)  # Transpose to shape (784, hidden_dim)
        
        # Trainable supervised weights
        n_hidden = W_unsupervised.size(0)  # Assuming W_unsupervised now has shape (784, hidden_dim)
        self.S = nn.Linear(n_hidden, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):

        # Forward through unsupervised weights
        x = torch.matmul(x, self.W_unsupervised)  # Resulting shape should be (batch_size, hidden_dim)
        # Forward through supervised weights
        x = self.S(x)
        x = self.bn(x)
        return self.softmax(x)


def train_bio_classifier(W_unsupervised, train_loader, val_loader, correct_img_bzs, data_name, n_hidden, out_features, n_epochs, batch_size, anti_hebbian_learning_strength, 
                       lebesgue_norm, rank):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clone initial unsupervised weights for verification
    W_initial = W_unsupervised.clone().detach()
    W_unsupervised.requires_grad = False

    # Initialize BioClassifier with frozen W_unsupervised
    model = BioClassifier(W_unsupervised, out_features).to(device)
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.S.parameters(), lr=0.001)
    
    # Scheduler setup
    scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200, 250], gamma=0.5)
    
    # Lists for logging
    train_loss_log, val_loss_log, train_acc_log, val_acc_log = [], [], [], []
    
    for epoch in range(n_epochs):
        model.train()
        total_loss, correct = 0, 0
        
        for images, labels in train_loader:
            if images.shape[0] != correct_img_bzs:
                print(f"Batch scartato per dimensione errata: {images.shape}")
                continue

            images = images.view(batch_size, -1).to(device)  # Flatten images to (batch_size, img_sz)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        
        # Store training logs
        train_loss_log.append(train_loss)
        train_acc_log.append(train_accuracy)

        # Validation every epochs
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_acc:.4f}")

        # Scheduler step
        scheduler.step()
    
    # Verify unsupervised weights remain unchanged
    if torch.equal(W_unsupervised, W_initial):
        print("Unsupervised weights remained unchanged during supervised training.")
    else:
        print("Warning: Unsupervised weights were modified during supervised training.")
    
    
    # Plotting
    # Dizionario dei nomi dei dataset
    data_name_dict = {1: "MNIST", 2: "CIFAR10", 3: "FashionMNIST"}
    dataset_name = data_name_dict.get(data_name, "UnknownDataset")

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.plot(np.linspace(0, len(val_loss_log), len(val_loss_log)), train_loss_log, '.-', label='Train Loss', color='red', lw=2, alpha=0.5)
    plt.plot(np.linspace(0, len(val_loss_log), len(val_loss_log)), val_loss_log, '.-', label='Val Loss', color='blue', lw=2, alpha=0.5)
    plt.title(f'Training and Validation Loss for {n_hidden} hidden units on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'output/{dataset_name}_biolinear_{n_hidden}hu_{out_features}out_{n_epochs}ep_ahls{anti_hebbian_learning_strength}_lnorm{lebesgue_norm}_rank{rank}_loss.png')
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(np.linspace(0, len(val_loss_log), len(val_loss_log)), train_acc_log, '.-', label='Train Accuracy', color='red', lw=2, alpha=0.5)
    plt.plot(np.linspace(0, len(val_loss_log), len(val_loss_log)), val_acc_log, '.-', label='Val Accuracy', color='blue', lw=2, alpha=0.5)
    plt.title(f'Training and Validation Accuracy for {n_hidden} hidden units on {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'output/{dataset_name}_biolinear_{n_hidden}hu_{out_features}out_{n_epochs}ep_ahls{anti_hebbian_learning_strength}_lnorm{lebesgue_norm}_rank{rank}_acc.png')
    plt.show()

    return train_loss_log, val_loss_log, train_acc_log, val_acc_log


def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    # Disable gradient calculations for validation
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(images.size(0), -1).to(device)  # Flatten images and move to device
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy
