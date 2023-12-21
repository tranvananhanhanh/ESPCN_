from torch.utils.data import random_split
from dataset import train_loader, val_loader  
from torch.utils.data import DataLoader
from model import model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

def PSNR(prediction, target, max_pixel=1.0):
    mse = torch.mean((prediction - target) ** 2)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
train_loss_history = []
train_psnr_history = []
val_loss_history = []
val_psnr_history = []

for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0
    total_psnr = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_psnr = PSNR(outputs, targets)
        total_psnr += batch_psnr.item()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    average_psnr = total_psnr / len(train_loader)
    train_loss_history.append(average_loss)
    train_psnr_history.append(average_psnr)

    print(f'Training - Epoch {epoch + 1}/{num_epochs}, Avg. Loss: {average_loss}, Avg. PSNR: {average_psnr}')

    model.eval()  
    with torch.no_grad():
        val_total_loss = 0.0
        val_total_psnr = 0.0

        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_batch_psnr = PSNR(val_outputs, val_targets)

            val_total_loss += val_loss.item()
            val_total_psnr += val_batch_psnr.item()

        val_average_loss = val_total_loss / len(val_loader)
        val_average_psnr = val_total_psnr / len(val_loader)
        val_loss_history.append(val_average_loss)
        val_psnr_history.append(val_average_psnr)

        
        print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Avg. Loss: {val_average_loss}, Avg. PSNR: {val_average_psnr}')


torch.save(model.state_dict(), 'my_model.pth')

plt.figure(figsize=(12, 4))


plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_psnr_history, label='Training PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(val_psnr_history, label='Validation PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()

plt.tight_layout()
plt.show()
