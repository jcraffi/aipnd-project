import argparse
import subprocess
import time
from supported_models import model_info

def run_training_script(args, model_name):
    command = [
        'python', 'train.py', args.data_dir,
        '--save_dir', args.save_dir,
        '--arch', model_name,
        '--learning_rate', str(args.learning_rate),
        '--hidden_units', str(args.hidden_units),
        '--epochs', str(args.epochs),
        '--subset', str(args.subset)
    ]
    if args.gpu:
        command.append('--gpu')

    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    total_time = end_time - start_time
    test_loss = None
    test_accuracy = None

    for line in result.stdout.split('\n'):
        if 'Test Loss' in line and 'Test Accuracy' in line:
            parts = line.split('..')
            test_loss = parts[0].split(':')[-1].strip()
            test_accuracy = parts[1].split(':')[-1].strip()

    return model_name, total_time, test_loss, test_accuracy

def main(): # Example code to run: python model_comparison.py 'flowers' --save_dir checkpoint --gpu --subset .1 --learning_rate 0.001
    parser = argparse.ArgumentParser(description='Run training for each model and output results')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--subset', type=float, default=None, help='Subset of the dataset to use for training')
    args = parser.parse_args()

    results = []
    for model_name in model_info.keys():
        model_name, total_time, test_loss, test_accuracy = run_training_script(args, model_name)
        results.append((model_name, total_time, test_loss, test_accuracy))

    with open('model_comparison_results.txt', 'w') as f:
        for model_name, total_time, test_loss, test_accuracy in results:
            f.write(f"Model: {model_name}\n")
            f.write(f"Total Training Time: {total_time:.2f} seconds\n")
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")
            f.write("\n")

    print("Model comparison results have been written to model_comparison_results.txt")

if __name__ == '__main__':
    main()