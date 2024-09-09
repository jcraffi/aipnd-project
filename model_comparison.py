import time
import subprocess
import torch
from supported_models import model_info
from train_and_predict_utils import test_model
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def main(): # Example code to run: python train.py 'flowers' --save_dir checkpoint --epochs 10 --gpu --subset 100
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    # Add argument to subset the data for faster training on small GPUs and CPUs
    parser.add_argument('--subset', type=int, default=None, help='Subset of the dataset to use for training')
    args = parser.parse_args()

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

    _ , _ , test_loader = load_data(args.data_dir, args.subset)


    # Open the results file
    with open('model_comparison_results.txt', 'w') as f:
        for model_name, model_params in model_info.items():
            # Prepare the command to run train.py with the necessary arguments
            command = [
                'python', 'train.py', data_dir,
                '--arch', model_name,
                '--save_dir', 'checkpoint',
                '--epochs', '10',
                '--gpu',
                '--subset', '75'
            ]

            # Measure the time and run the command
            start_time = time.time()
            result = subprocess.run(command, capture_output=True, text=True)
            end_time = time.time()

            # Calculate the training time
            training_time = end_time - start_time

            # Extract training accuracy from the output (assuming train.py prints it)
            output_lines = result.stdout.split('\n')
            training_accuracy = None
            for line in output_lines:
                if 'Training Accuracy:' in line:
                    training_accuracy = line.split('Training Accuracy:')[-1].strip()
                    break

            # Load the trained model
            model = torch.load('checkpoint.pth')
            model.to(device)

            # Test the model
            test_loss, test_accuracy = test_model(model, test_loader, criterion, device)

            # Write the results to the file
            f.write(f"Model: {model_name}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            if training_accuracy:
                f.write(f"Training Accuracy: {training_accuracy}\n")
            f.write(f"Test Loss: {test_loss:.2f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
            f.write("\n")

    print("Model comparison results have been written to model_comparison_results.txt")