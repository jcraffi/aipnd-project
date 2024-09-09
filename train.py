import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from data_utils import load_data
from train_utils import train_model, save_checkpoint

# Example code to run: python train.py 'flowers' --save_dir checkpoint.pth --arch vgg16 --epochs 10 --gpu

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() 
                          else "mps" if args.gpu and torch.backends.mps.is_available()
                          else "cpu")
    
    trainloader, validloader, _ = load_data(args.data_dir)
    
    weights = VGG16_Weights.IMAGENET1K_V1 if args.arch == 'vgg16' else None
    model = getattr(models, args.arch)(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(args.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    model.to(device)
    
    train_model(model, trainloader, validloader, criterion, optimizer, device, args.epochs)
    
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()