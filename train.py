import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from train_and_predict_utils import load_data, create_model, test_model, train_model, save_checkpoint
from os import makedirs 
from supported_models import model_info


def main(): # Example code to run: python train.py 'flowers' --save_dir checkpoint --arch vgg13 --epochs 10 --gpu --subset .1 --learning_rate 0.001
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    # Add argument to subset the data for faster training on small GPUs and CPUs
    parser.add_argument('--subset', type=float, default=None, help='Subset of the dataset to use for training')
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
    train_loader, valid_loader, test_loader, class_to_idx, cat_to_name = load_data(args.data_dir, args.subset)
    
    # Get the model information based on the architecture
    model, criterion, optimizer = create_model(args.arch, device, class_to_idx, cat_to_name, args.learning_rate, args.hidden_units)

    # Load the model with the corresponding weights
    model, train_losses, valid_losses = train_model(model, criterion, optimizer, train_loader, valid_loader, args.epochs, device)

    # Test the model
    test_loss, accuracy = test_model(model, criterion, test_loader, device)

    # Save the model as a checkpoint
    save_checkpoint(args.arch, model, optimizer, args.epochs, args.learning_rate, args.hidden_units, args.save_dir)

if __name__ == '__main__':
    main()