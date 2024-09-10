import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from train_and_predict_utils import load_data, train_model, save_checkpoint
from os import makedirs 
from supported_models import model_info


def main(): # Example code to run: python train.py 'flowers' --save_dir checkpoint --arch vgg16 --epochs 10 --gpu --subset 100
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    # Add argument to subset the data for faster training on small GPUs and CPUs
    parser.add_argument('--subset', type=int, default=None, help='Subset of the dataset to use for training')
    args = parser.parse_args()
    
    #Assign the device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() 
                          else "mps" if args.gpu and torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")
    
    # Validate the model architecture
    model_names = list(model_info.keys())
    if args.arch not in list(model_info.keys()):
        raise ValueError(f"Unsupported architecture '{args.arch}'. Choose from {model_names}")
    
    # Create the save directory if it does not exist
    makedirs(args.save_dir, exist_ok=True)

    # Load the data
    train_loader, valid_loader, _ = load_data(args.data_dir, args.subset)
    
    # Get the model information based on the architecture
    info = model_info.get(args.arch, {
        'weights': None,
        'in_features': lambda model: None,
        'criterion': nn.NLLLoss,
        'optimizer': lambda model, lr: optim.Adam(model.parameters(), lr=lr)
    })

    # Load the model with the corresponding weights
    model = getattr(models, args.arch)(weights=info['weights'])
    for param in model.parameters():
        param.requires_grad = False

    # Extract the in_features using the corresponding function
    in_features = info['in_features'](model)

    # Define the classifier
    model.classifier = nn.Sequential(
        nn.Linear(in_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # Define the criterion and optimizer based on the model
    criterion = info['criterion']()
    optimizer = info['optimizer'](model, lr=args.learning_rate)
        
    model.to(device)
    
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

    # Access the original dataset to get class_to_idx
    if isinstance(train_loader.dataset, torch.utils.data.Subset):
        original_dataset = train_loader.dataset.dataset
    else:
        original_dataset = train_loader.dataset

    class_to_idx = original_dataset.class_to_idx

    save_checkpoint(model, args.save_dir, args.arch, class_to_idx, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()