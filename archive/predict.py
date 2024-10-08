import argparse
import torch
from train_and_predict_utils import load_checkpoint, predict

def main(): # Example code to run: python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoint' --category_names cat_to_name.json --gpu
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str,  help='Path to category to names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() 
                          else "mps" if args.gpu and torch.backends.mps.is_available()
                          else "cpu")
    
    #load the model
    model, optimizer = load_checkpoint(args.checkpoint)
    
    # Get the top K probabilities and indices
    predict(args.input, model, device, args.top_k)

if __name__ == '__main__':
    main()