import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging
from model import MLP  # Import the MLP model class
from data_loader import load_data  # Import the custom data loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Setup logging for displaying training information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """
    Train the MLP model on the Visionline dataset. This function splits the data into training
    and testing sets, then iteratively trains the model using the training set while evaluating
    it on the test set for validation.
    
    The model's weights are saved to a file after training.
    """

    # Load the dataset
    dataset = load_data('/home/elnaz/datasets/visionline')

    # Split dataset into training and testing (80% training, 20% testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Initialize the MLP model (input size: 3*128*128 for RGB images)
    model = MLP(3 * 128 * 128)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    # Create data loaders for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle the training data
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   # Don't shuffle test data

    # Set the number of training epochs
    num_epochs = 50  # Number of epochs for training

    # Lists to store loss and accuracy for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_train = 0
        total_train = 0

        # Training Phase
        for features, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients from the previous step
            outputs = model(features.view(features.size(0), -1))  # Flatten the input image
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model's parameters
            epoch_loss += loss.item()  # Accumulate the training loss

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
            total_train += labels.size(0)  # Total number of training samples
            correct_train += (predicted == labels).sum().item()  # Count correct predictions

        # Calculate and store average training loss and accuracy
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        train_acc = correct_train / total_train
        train_accuracies.append(train_acc)

        # Validation Phase (evaluation on test data)
        model.eval()  # Set the model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss = 0
        with torch.no_grad():  # No gradient calculation for validation
            for features, labels in test_loader:
                outputs = model(features.view(features.size(0), -1))  # Flatten input for the model
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()  # Accumulate validation loss

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
                total_val += labels.size(0)  # Total number of validation samples
                correct_val += (predicted == labels).sum().item()  # Count correct predictions

        # Calculate and store average validation loss and accuracy
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        val_acc = correct_val / total_val
        val_accuracies.append(val_acc)

        # Log the training and validation results for the current epoch
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, '
                    f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # Save the trained model's weights to a file
    torch.save(model.state_dict(), 'model.pt')
    logger.info('Model training complete and saved as model.pt')

if __name__ == "__main__":
    train_model()  # Call the train_model function if the script is run directly
