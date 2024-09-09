import torch

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

        
def predict(image_path, model, device, topk=5):
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    probs, classes = torch.exp(output).topk(topk)
    return probs.cpu().numpy()[0], classes.cpu().numpy()[0]