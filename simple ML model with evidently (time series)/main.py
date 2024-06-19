import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric, ColumnDistributionMetric
import prometheus_client
from statHook import StatHook
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific NumPy runtime warnings
np.seterr(divide='ignore', invalid='ignore')


def start_prometheus_server():
    try:
        prometheus_client.start_http_server(8000)
    except Exception as e:
        logger.error(f"Error starting Prometheus server: {e}")


class TitanicMLP(nn.Module):
    def __init__(self, input_size):
        super(TitanicMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    data = data[features + ['Survived']]
    fill_values = {'Age': data['Age'].median(), 'Fare': data['Fare'].median(), 'Embarked': 'S'}
    data.fillna(value=fill_values, inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data, features


def prepare_tensors(data, features):
    X = data[features].values
    stds = X.std(axis=0)
    means = X.mean(axis=0)
    X_normalized = (X - means) / (stds + 1e-8)  # Adding epsilon to avoid divide by zero
    if np.any(np.isnan(X_normalized)) or np.any(np.isinf(X_normalized)):
        logger.warning("NaNs or infinite values found in normalized data.")

    y = data['Survived'].values
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def prometheus_metrics():
    training_mean = prometheus_client.Summary('layer_output_train_mean', 'Mean of the layer outputs')
    training_std = prometheus_client.Summary('layer_output_train_std', 'Std of the layer outputs')
    training_distribution = prometheus_client.Histogram('layer_output_train_distribution',
                                                        'distribution of the layer outputs',
                                                        buckets=np.linspace(-3, 3, 21).tolist())
    testing_mean = prometheus_client.Summary('layer_output_test_mean', 'Mean of the layer outputs during testing')
    testing_std = prometheus_client.Summary('layer_output_test_std', 'Std of the layer outputs during testing')
    testing_distribution = prometheus_client.Histogram('layer_output_test_distribution',
                                                       'distribution of the layer outputs during testing',
                                                       buckets=np.linspace(-3, 3, 21).tolist())
    return training_mean, training_std, training_distribution, testing_mean, testing_std, testing_distribution


def run_model():
    data, features = load_data()
    X_tensor, y_tensor = prepare_tensors(data, features)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    num_cycles = 5  # Number of reshuffling cycles
    num_epochs = 10  # Number of epochs per cycle
    all_means = []
    all_stds = []
    all_distributions = []

    training_mean, training_std, training_distribution, testing_mean, testing_std, testing_distribution = prometheus_metrics()
    model = TitanicMLP(len(features))
    stat_hook = StatHook()
    hook_handle = model.fc2.register_forward_hook(stat_hook.hook_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for cycle in range(num_cycles):
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        model.train()
        for epoch in range(num_epochs):
            for features, labels in torch.utils.data.DataLoader(train_dataset, batch_size=64):
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if stat_hook.means and stat_hook.stds:
                    current_mean = stat_hook.means[-1].item()
                    current_std = stat_hook.stds[-1].item()
                    current_distribution = stat_hook.hist_data[-1]

                    all_means.append(current_mean)
                    all_stds.append(current_std)

                    epoch_cycle = epoch + cycle * num_epochs
                    training_mean.observe(current_mean)
                    training_std.observe(current_std)

                    all_distributions.extend(current_distribution.flatten().tolist())

            logger.info(f'Cycle {cycle + 1}, Epoch {epoch + 1}, Loss: {loss.item()}')

    training_data = pd.DataFrame({'means': all_means, 'stds': all_stds, 'distributions': np.repeat(all_distributions, len(all_means) // len(all_distributions) + 1)[:len(all_means)]})

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    predictions, means, stds, distributions = [], [], [], []
    for features, _ in test_loader:
        with torch.no_grad():
            single_input = features.unsqueeze(0)
            output = model(features)
            stat_hook.hook_fn(model.fc2, single_input, output)

            if stat_hook.means and stat_hook.stds:
                current_mean = stat_hook.means[-1]
                current_std = stat_hook.stds[-1]
                current_distribution = stat_hook.hist_data[-1]

                means.append(current_mean)
                stds.append(current_std)
                distributions.extend(current_distribution.flatten().tolist())

                testing_mean.observe(current_mean)
                testing_std.observe(current_std)
                for activation in current_distribution.flatten():
                    testing_distribution.observe(activation)

                prediction = torch.argmax(output, dim=1).item()
                predictions.append(prediction)

        pred = torch.argmax(output, dim=1)
        predictions.append(pred.item())
        means.append(output.mean().item())
        stds.append(output.std().item())

    test_data = pd.DataFrame({'means': means, 'stds': stds, 'predictions': predictions, 'distributions': np.repeat(distributions, len(means) // len(distributions) + 1)[:len(means)]})
    return training_data, test_data


def generate_reports(typ, data):
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    data['means'] = data['means'].astype(float)
    data['stds'] = data['stds'].astype(float)
    data['distributions'] = data['distributions'].astype(float)


    # if 'timestamp' not in data.columns:
    #     # Create a synthetic timestamp if not available
    #     total_duration = pd.Timedelta(hours=24)  # Example: Cover an hour
    #     interval = total_duration / len(data)
    #     data['timestamp'] = pd.date_range(end=datetime.now(), periods=len(data), freq=interval)
    #
    # data.set_index('timestamp', inplace=True)

    if typ == 'training':
        reference_mean = data['means'].mean()
        reference_std = data['stds'].mean()
        reference_distributions = data['distributions'].mean()
        reference_data = pd.DataFrame({'means': [reference_mean] * len(data), 'stds': [reference_std] * len(data),
                                       'distributions': [reference_distributions] * len(data)}, index=data.index)
    else:
        reference_mean = data['means'].mean()
        reference_std = data['stds'].mean()
        reference_distributions = data['distributions'].mean()
        reference_data = pd.DataFrame({'means': [reference_mean] * len(data), 'stds': [reference_std] * len(data),
                                       'predictions': [data['predictions'].mode()[0]] * len(data),
                                       'distributions': [reference_distributions] * len(data)}, index=data.index)
        data['predictions'] = data['predictions'].astype(int)

    reference_data = reference_data.apply(pd.to_numeric, errors='coerce')

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=data)
    name_html = f'data_drift_{typ}_report.html'
    report.save_html(name_html)
    logger.info(f"Evidently Data Drift Report is generated and saved as {name_html}.")


def main():
    start_prometheus_server()
    training_data, test_data = run_model()
    generate_reports('training', training_data)
    generate_reports('testing', test_data)


if __name__ == "__main__":
    main()
