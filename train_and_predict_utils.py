import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import random
from supported_models import model_info

# Create a function to load the data
def load_data(data_dir, subset_percentage=None):
    # Define the directories to use
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.5)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    common_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=common_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=common_transforms)

    print(f"Number of training images: {len(train_datasets)}")

    # Create class to index mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f,strict=False)
    class_to_idx = train_datasets.class_to_idx

    if subset_percentage is not None:
        train_size = int(len(train_datasets) * subset_percentage)
        valid_size = int(len(valid_datasets) * subset_percentage)
        
        train_indices = random.sample(range(len(train_datasets)), train_size)
        valid_indices = random.sample(range(len(valid_datasets)), valid_size)
        
        train_datasets = Subset(train_datasets, train_indices)
        valid_datasets = Subset(valid_datasets, valid_indices)

    print(f"Number of training images after subsetting: {len(train_datasets)}")
    
    # Define the dataloaders
    train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=32, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx, cat_to_name

# Define the model, criterion, and optimizer based on the model name and model info
def create_model(modal_name, lr, hidden_units):
    model = getattr(models, model_name)(weights=model_info[model_name]['weights'])
    model.class_to_idx = class_to_idx
    model.cat_to_name = cat_to_name

    # Freeze the feature extractor parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # Define the classifier
    model.classifier = nn.Sequential(
        nn.Linear(model_info[model_name]['in_features'](model), hidden_units), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.2),
        nn.Linear(hidden_units, 102), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.5),
        nn.LogSoftmax(dim=1)
    )
    model.to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)

    criterion = model_info[model_name]['criterion']()
    optimizer = model_info[model_name]['optimizer'](model, lr)
    return model, criterion, optimizer

# Define the training function
def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, device):
    train_losses, valid_losses = [], []
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.amp.autocast(device_type=device.type):
                    output = model(images)
                    loss = criterion(output, labels)
            else:
                output = model(images)
                loss = criterion(output, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels).item()
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss / len(train_loader))
            valid_losses.append(valid_loss / len(valid_loader))

            print(f"Epoch: {e+1}/{epochs}.. "
                f"Training Loss: {running_loss / len(train_loader):.3f}.. "
                f"Validation Loss: {valid_loss / len(valid_loader):.3f}.. "
                f"Validation Accuracy: {accuracy / len(valid_loader):.3f}")
        
        # Step the scheduler
        scheduler.step()
        
    return model, train_losses, valid_losses

# Test the network on the test data
def test_model(model, criterion, test_loader, device):
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
            

    print(f"Test Loss: {test_loss / len(test_loader):.3f}.. "
        f"Test Accuracy: {accuracy / len(test_loader) * 100:.2f}%")
    
    return test_loss, accuracy

# Save the model checkpoint
def save_checkpoint(model_name, model, optimizer, epochs, save_dir):
    # Create a dictionary to save the model state and additional information
    checkpoint = {
        'model_name': model_name,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    # Save the checkpoint
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

    print("Model and additional information saved successfully.")

# Load the model checkpoint
def load_checkpoint(save_dir):
    checkpoint = torch.load(f'{save_dir}/checkpoint.pth')

    model = getattr(models, checkpoint['model_name'])(weights=model_info[checkpoint['model_name']]['weights'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])
    class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    ###
    # Load the model

    # Load the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Print confirmation message
    print("Model and additional information loaded successfully.")
    
    # Return the model, optimizer, and class_to_idx
    return model, optimizer, class_to_idx

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

def predict(image_path, model, device, topk=5):
    # Process the image and add batch dimension
    image = process_image(image_path).unsqueeze(0).to(device)
    
    # Set the model to evaluation mode and send it to the device
    model.eval().to(device)
    
    # Turn off gradients for inference
    with torch.no_grad():
        # Forward pass
        output = model(image)
        
    # Get the probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the topk probabilities and indices
    topk_probabilities, topk_indices = probabilities.topk(topk)
    ###
    # Convert to lists
    topk_probabilities = topk_probabilities.tolist()
    topk_indices = topk_indices.tolist()
    ###
    return topk_probabilities, topk_indices

def display_predictions(model, topk_probabilities, topk_indices, class_names, topk=5):
    # Map indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in topk_indices]
    
    # Get the class names
    topk_class_names = [class_names[str(cls)] for cls in top_classes]

    # Print the top K predictions
    for cls, prob in zip(topk_class_names, topk_probabilities):
        print(f"Class: {cls:<20} Probability: {prob * 100:>6.1f}%")
