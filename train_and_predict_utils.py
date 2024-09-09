import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import random
from supported_models import model_info

def load_data(data_dir,subset=None):
    train_dir = f'{data_dir}/train'
    valid_dir = f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    if subset is not None:
        train_indices = random.sample(range(len(train_datasets)), subset)
        train_datasets = Subset(train_datasets, train_indices)
        valid_indices = random.sample(range(len(valid_datasets)), subset)
        valid_datasets = Subset(valid_datasets, valid_indices)
    
    train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=32)
    test_loader = DataLoader(test_datasets, batch_size=32)
    
    return train_loader, valid_loader, test_loader

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def train_model(model, trainloader, validloader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        scaler = torch.amp.GradScaler('cuda')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                output = model(images)
                loss = criterion(output, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch: {epoch+1}/{epochs}.. "
              f"Training Loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation Accuracy: {accuracy/len(validloader):.3f}")
    
def test_model(model, test_loader, criterion, device):
    # Test the network on the test data
    model.eval()
    test_loss = 0
    accuracy = 0

    # Disable gradient calculation for testing
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels).item()
            
            # Calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return test_loss, accuracy

def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(weights=model_info[checkpoint['arch']])

    # Find the first Linear layer in the classifier
    in_features = None
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            break

    if in_features is None:
        raise ValueError("No Linear layer found in the classifier")

    model.classifier = nn.Sequential(
        nn.Linear(in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open the image
    image = Image.open(image_path)
    
    # Apply the transformations
    tensor = transform(image)
    
    return tensor

def predict(image_path, model, device, topk=5):
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    probs, classes = torch.exp(output).topk(topk)
    return probs.cpu().numpy()[0], classes.cpu().numpy()[0]