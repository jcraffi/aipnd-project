import argparse
import time
import subprocess
import torch
from supported_models import model_info
from train_and_predict_utils import process_image, test_model
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def run_training_script(args):
    command = [
        'python', 'train.py', args.data_dir,
        '--save_dir', args.save_dir,
        '--arch', args.arch,
        '--learning_rate', str(args.learning_rate),
        '--hidden_units', str(args.hidden_units),
        '--epochs', str(args.epochs),
        '--subset', str(args.subset)
    ]
    if args.gpu:
        command.append('--gpu')

    result = subprocess.run(command, capture_output=True, text=True)
    return result

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Script')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--subset', type=int, default=100, help='Subset of data to use for training')
    
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() 
                        else "mps" if args.gpu and torch.backends.mps.is_available()
                        else "cpu")
    print(f"Device: {device}")

    with open('model_comparison_results.txt', 'w') as f:
        for model_name, info in model_info.items():
            start_time = time.time()
            
            # Train the model
            args.arch = model_name
            result = run_training_script(args)
            training_time = time.time() - start_time
            
            # Extract training accuracy from the output (assuming train.py prints it)
            output_lines = result.stdout.split('\n')
            training_accuracy = None
            for line in output_lines:
                if 'Training Accuracy:' in line:
                    training_accuracy = line.split('Training Accuracy:')[-1].strip()
                    break
            
            # Load the trained model
            checkpoint = torch.load(f'{args.save_dir}/checkpoint.pth')
            model = models.__dict__[model_name](pretrained=False)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)

            # Test the model
            test_start_time = time.time()
            test_loader = DataLoader(datasets.ImageFolder(f'{args.data_dir}/test', transform=process_image))
            test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
            test_time = time.time() - test_start_time
            
            total_time = training_time + test_time
            
            # Write the results to the file
            f.write(f"Model: {model_name}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            if training_accuracy:
                f.write(f"Training Accuracy: {training_accuracy}\n")
            f.write(f"Test Loss: {test_loss:.2f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
            f.write(f"Test Time: {test_time:.2f} seconds\n")
            f.write(f"Total Time: {total_time:.2f} seconds\n")
            f.write("\n")
    
    print("Model comparison results have been written to model_comparison_results.txt")

if __name__ == '__main__':
    main()