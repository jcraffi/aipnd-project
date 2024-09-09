import time
import subprocess
from supported_models import model_info

# Open the results file
with open('model_comparison_results.txt', 'w') as f:
    for model_name, model_params in model_info.items():
        # Prepare the command to run train.py with the necessary arguments
        command = [
            'python', 'train.py', 'flowers',
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

        # Write the results to the file
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        if training_accuracy:
            f.write(f"Training Accuracy: {training_accuracy}\n")
        f.write("\n")

print("Model comparison results have been written to model_comparison_results.txt")