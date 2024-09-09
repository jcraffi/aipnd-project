import torch
from torch.cuda.amp import GradScaler, autocast

def train_model(model, trainloader, validloader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                output = model(images)
                loss = criterion(output, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Epoch: {epoch+1}/{epochs}.. "
              f"Training Loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation Accuracy: {accuracy/len(validloader):.3f}")
    
def save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
