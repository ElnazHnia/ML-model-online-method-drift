# Import the necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
import prometheus_client
from statHook import StatHook

# Start Prometheus HTTP server
prometheus_client.start_http_server(8000)

# Define Prometheus gauges for monitoring
mean_gauge = prometheus_client.Gauge('layer_output_mean', 'Mean of the layer outputs')
std_dev_gauge = prometheus_client.Gauge('layer_output_std_dev', 'Standard deviation of the layer outputs')

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Select relevant features and handle missing values
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
titanic = titanic[features + ['Survived']]

# Fill missing values using a dictionary
fill_values = {
    'Age': titanic['Age'].median(),
    'Fare': titanic['Fare'].median(),
    'Embarked': 'S'
}
titanic.fillna(value=fill_values, inplace=True)
# Convert categorical variables to numeric
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Normalize the feature data
X = titanic[features].values
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(titanic['Survived'].values, dtype=torch.long)

# Split into training and testing sets (80/20)
train_size = int(0.8 * len(X_tensor))
test_size = len(X_tensor) - train_size
X_train, X_test = torch.utils.data.random_split(X_tensor, [train_size, test_size])
y_train, y_test = torch.utils.data.random_split(y_tensor, [train_size, test_size])


# Function to convert Subset to Tensor
def subset_to_tensor(subset):
    return torch.stack([subset.dataset[i] for i in subset.indices])


# Convert Subset to Tensor for viewing
X_train_tensor = subset_to_tensor(X_train)
y_train_tensor = subset_to_tensor(y_train)
X_test_tensor = subset_to_tensor(X_test)
y_test_tensor = subset_to_tensor(y_test)


# Define the MLP model
class TitanicMLP(nn.Module):
    def __init__(self):
        super(TitanicMLP, self).__init__()
        self.fc1 = nn.Linear(len(features), 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
model = TitanicMLP()

# Instantiate StatHook and register the hook
stat_hook = StatHook()
hook_handle = model.fc2.register_forward_hook(stat_hook.hook_fn)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 10
num_cycles = 5  # Number of times to reshuffle and re-split the dataset
for cycle in range(num_cycles):
    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Prometheus gauges with the latest hook statistics
        if stat_hook.means and stat_hook.stds:
            current_mean = stat_hook.means[-1]
            current_std = stat_hook.stds[-1]
            mean_gauge.set(current_mean)
            std_dev_gauge.set(current_std)
            print(f'Mean: {current_mean}, STD: {current_std}')

        if (epoch + 1) % 10 == 0:
            print(f'Cycle {cycle + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Reshuffle and re-split the dataset after each full cycle of epochs
    indices = torch.randperm(len(X_tensor))
    X_tensor, y_tensor = X_tensor[indices], y_tensor[indices]
    X_train, X_test = torch.utils.data.random_split(X_tensor, [train_size, test_size])
    y_train, y_test = torch.utils.data.random_split(y_tensor, [train_size, test_size])

    X_train_tensor = subset_to_tensor(X_train)
    y_train_tensor = subset_to_tensor(y_train)
    X_test_tensor = subset_to_tensor(X_test)
    y_test_tensor = subset_to_tensor(y_test)

# Remove the hook
hook_handle.remove()

# Test the model
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    _, predicted_test = torch.max(y_test_pred, 1)
    accuracy_test = (predicted_test == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy_test:.4f}')


# Create Evidently Report
# print("Generating report...")
#
# column_mapping = ColumnMapping(
#     target='target',
#     prediction='prediction',
#     numerical_features=features
# )
# Create Evidently Report
# report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), ClassificationPreset()])
# train_data = pd.DataFrame(X_train_tensor.numpy(), columns=features)
# train_data['prediction'] = predicted_test.numpy()
# train_data['target'] = y_test_tensor.numpy()
# test_data = pd.DataFrame(X_test_tensor.numpy(), columns=features)
# test_data['prediction'] = predicted_test.numpy()
# test_data['target'] = y_test_tensor.numpy()
# report.run(reference_data=train_data, current_data=test_data, column_mapping=column_mapping)
# report.save_html('model_monitoring_report.html')
# print("Report generated and saved.")
