import argparse
import torch
from torchvision import models
import json
from train_and_predict_utils import load_checkpoint, predict

def main(): # Example code to run: python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoint/checkpoint.pth' --category_names cat_to_name.json --gpu
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
    
    model, class_to_idx = load_checkpoint(args.checkpoint)
    model.to(device)
    
    probs, classes = predict(args.input, model, device, args.top_k)

    # Reverse the class_to_idx dictionary to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Get the class labels from the predicted indices
    class_labels = [idx_to_class[idx] for idx in classes]
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f, strict=False)
        class_labels = [cat_to_name.get(str(cls), "Unknown") for cls in class_labels]
    
    for cls, prob in zip(class_labels, probs):
        print(f"Class: {cls:<20} Probability: {prob * 100:>6.1f}%")

if __name__ == '__main__':
    main()