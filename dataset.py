import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split

class MyBlurredDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_paths = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.blur_transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=15),  
            
        ])
        self.downsample_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        
        blurred_image = self.blur_transform(image)

        
        downsampled_image = self.downsample_transform(blurred_image)
        image=transforms.ToTensor()(image)
        return downsampled_image, image


image_path = "/Users/jmac/Desktop/ESPCN/resizee"


blurred_dataset = MyBlurredDataset(image_path)

blurred_dataloader = DataLoader(blurred_dataset, batch_size=8, shuffle=True)

for downsampled_images, blurred_images in blurred_dataloader:
    print("LR images shape:", downsampled_images.shape)
    print("HR images shape:", blurred_images.shape)
    break  


total_size = len(blurred_dataloader.dataset)
train_size = int(0.6 * total_size)  # 60% for training
val_size = int(0.2 * total_size)    # 20% for validation
test_size = total_size - train_size - val_size  # Remaining 20% for testing

train_dataset, val_dataset, test_dataset = random_split(blurred_dataloader.dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
