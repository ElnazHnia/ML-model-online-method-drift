import torch.nn as nn
import torch

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for binary classification tasks.
    
    Args:
        input_size (int): The number of input features for the model.
        
    The model consists of three fully connected (Linear) layers:
    - fc1: Input layer that takes the input features and maps them to 10 hidden units.
    - fc2: Hidden layer with 10 input units and 8 output units.
    - fc3: Output layer with 2 output units for binary classification.
    
    Each hidden layer is followed by a ReLU activation function.
    """
    def __init__(self, input_size):
        super(MLP, self).__init__()
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(input_size, 10)  # First layer maps input_size to 10 units
        self.fc2 = nn.Linear(10, 8)           # Second layer maps 10 units to 8 units
        self.fc3 = nn.Linear(8, 2)            # Output layer maps 8 units to 2 (binary classification)

    def forward(self, x):
        """
        Defines the forward pass of the model. The input goes through two hidden layers
        with ReLU activation functions, and the final output is returned without activation.
        
        Args:
            x (Tensor): The input tensor to the model.
        
        Returns:
            Tensor: The output logits from the model.
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU after the first fully connected layer
        x = torch.relu(self.fc2(x))  # Apply ReLU after the second fully connected layer
        return self.fc3(x)           # Final output logits (no activation)
