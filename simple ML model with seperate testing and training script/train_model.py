import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from model import TitanicMLP
from data_loader import load_data, prepare_tensors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    data, features = load_data()
    X_tensor, y_tensor = prepare_tensors(data, features)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, test_size])

    model = TitanicMLP(len(features))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(10):  # Adjust the number of epochs as needed
        for features, labels in DataLoader(train_dataset, batch_size=64):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pt')
    logger.info('Model training complete and saved as model.pt')


if __name__ == "__main__":
    train_model()
