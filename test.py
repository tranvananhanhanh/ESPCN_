from dataset import test_loader
import torch
from train import model
from train import PSNR
import torch.nn as nn
from model import model
# Assuming your model is an instance of the ESPCN class

# Load the state dictionary
model.load_state_dict(torch.load('/Users/jmac/Desktop/ESPCN/my_model.pth'))

criterion = nn.MSELoss()

def evaluate(model, test_loader, criterion):
    model.eval()  # Switch to evaluation mode
    total_loss = 0
    total_psnr = 0

    with torch.no_grad():  # Turn off gradient calculation
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            psnr = PSNR(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            total_psnr += psnr.item() * inputs.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    avg_psnr = total_psnr / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}')

# Call the evaluate function
evaluate(model, test_loader, criterion)
