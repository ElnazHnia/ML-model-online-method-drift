import torch
import logging
import pandas as pd
import prometheus_client
from prometheus_client import CollectorRegistry, Summary, push_to_gateway
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import TitanicMLP
from data_loader import load_data, prepare_tensors
from statHook import StatHook
import numpy as np
import time
from driftDetection import DriftDetection
from torchsummary import summary
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_prometheus_server():
    try:
        prometheus_client.start_http_server(8000)
        logger.info("Prometheus server started on port 8000")
    except Exception as e:
        logger.error(f"Error starting Prometheus server: {e}")


# Setting the stattest for individual features
def set_stattest_individual_features():
    per_column_stattest = {x: 'wasserstein' for x in ['means', 'stds']}
    return per_column_stattest


def run_model():
    data, features = load_data()
    # Convert features to tensor format for prepare_tensors
    X_tensor, y_tensor = prepare_tensors(data, features)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    model = TitanicMLP(len(features))
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # Print model summary
    # print(f" len(features): {len(features)}")
    # input_size = (len(features),)  # Adjust input size based on your model
    # summary(model, input_size)

    stat_hook = StatHook()
    hook_handle = model.fc2.register_forward_hook(stat_hook.hook_fn)
    test_loader = DataLoader(test_dataset, batch_size=1)

    predictions, means, stds = [], [], []

    for features_t, _ in test_loader:
        with torch.no_grad():
            single_input = features_t.unsqueeze(0)
            output = model(features_t)
            stat_hook.hook_fn(model.fc2, single_input, output)

            if stat_hook.means and stat_hook.stds:
                current_mean = stat_hook.means[-1].item()
                current_std = stat_hook.stds[-1].item()

                means.append(current_mean)
                stds.append(current_std)

                # Get adjusted timestamp with time zone
                # training_mean_gauge.set(current_mean)
                # training_std_gauge.set(current_std)
                # time.sleep(15)  # 15

                # Check output tensor values
                # logger.info(f"Output tensor: {output}")

                prediction = torch.argmax(output, dim=1).item()
                # logger.info(f'features_t: {features_t}, Output: {output}, Prediction: {prediction}')
                predictions.append(prediction)

    # Introduce artificial drift in current data
    means[int(0.8 * len(means)):] = [x + 2 for x in means[int(0.8 * len(means)):]]
    # Ensure no NaN or inf values
    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    test_data = pd.DataFrame({'means': means, 'stds': stds, 'predictions': predictions})
    # test_data = pd.DataFrame({'means': means, 'stds': stds})

    '''
       Data drift detection 
    '''
    # Convert tensor data to pandas DataFrames
    train_features = torch.stack([x for x, y in train_dataset])
    test_features = torch.stack([x for x, y in test_dataset])
    train_df = pd.DataFrame(train_features.numpy(), columns=features)
    test_df = pd.DataFrame(test_features.numpy(), columns=features)

    '''
       Test Data Shows In Time Series
    '''
    DriftDetection().run_drift_detection(means, stds, predictions, test_data, train_df, test_df)
    return test_data


if __name__ == "__main__":
    start_prometheus_server()
    test_data = run_model()
    # time.sleep(60)
    # Keep the application running to serve metrics to Prometheus
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus server")
