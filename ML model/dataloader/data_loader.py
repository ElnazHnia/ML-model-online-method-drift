import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VisionlineDataset(Dataset):
    """
    A custom dataset class for loading images from two classes (0 and 1) stored in subdirectories.
    Each subdirectory corresponds to one class. Images are loaded, transformed, and prepared for 
    input into a deep learning model.
    
    Args:
        root_dir (str): Path to the root directory containing the class subdirectories.
        transform (callable, optional): Optional transform to be applied on a sample image.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Collect images and their labels from both class directories ('0' and '1')
        class_dirs = ['0', '1']
        for class_label in class_dirs:
            class_dir = os.path.join(root_dir, class_label)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.image_paths.append(file_path)
                self.labels.append(int(class_label))
        
        # Combine image paths and labels, then shuffle the data
        self.data = list(zip(self.image_paths, self.labels))
        random.shuffle(self.data)
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Fetches the image and its corresponding label at a given index.
        
        Args:
            idx (int): Index of the sample to be fetched.
        
        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Corresponding label of the image (0 or 1).
        """
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Open the image and convert it to RGB
        
        # Apply the transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data(root_dir):
    """
    Function to load and transform the dataset.
    
    Args:
        root_dir (str): Path to the root directory containing the dataset.
    
    Returns:
        VisionlineDataset: An instance of the custom VisionlineDataset with applied transformations.
    """
    # Define a series of transformations to be applied to each image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by 10 degrees
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image with mean and std
    ])
    
    # Create the dataset object
    dataset = VisionlineDataset(root_dir=root_dir, transform=transform)
    
    return dataset
