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
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_prometheus_server():
    try:
        prometheus_client.start_http_server(8000)
    except Exception as e:
        logger.error(f"Error starting Prometheus server: {e}")


def prometheus_metrics():
    testing_mean = prometheus_client.Histogram('layer_output_test_mean', 'Mean of the layer outputs during testing')
    testing_std = prometheus_client.Summary('layer_output_test_std', 'Std of the layer outputs during testing')
    testing_distribution = prometheus_client.Histogram('layer_output_test_distribution',
                                                       'distribution of the layer outputs during testing',
                                                       buckets=np.linspace(-3, 3, 21).tolist())
    return testing_mean, testing_std, testing_distribution


def run_model():
    data, features = load_data()
    X_tensor, y_tensor = prepare_tensors(data, features)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])

    testing_mean, testing_std, testing_distribution = prometheus_metrics()
    model = TitanicMLP(len(features))
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    stat_hook = StatHook()
    hook_handle = model.fc2.register_forward_hook(stat_hook.hook_fn)
    test_loader = DataLoader(test_dataset, batch_size=1)
    predictions, means, stds, distributions = [], [], [], []

    registry = CollectorRegistry()
    testing_mean_summary = Summary('layer_output_test1_mean', 'Mean of the layer outputs during testing',
                                   registry=registry)
    testing_std_summary = Summary('layer_output_test1_std', 'Std of the layer outputs during testing', registry=registry)

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


                testing_mean_summary.observe(current_mean)
                testing_std_summary.observe(current_std)
                push_to_gateway('pushgateway:9091', job='main', registry=registry)


                testing_mean.observe(current_mean)
                testing_std.observe(current_std)
                for activation in current_distribution.flatten():
                    testing_distribution.observe(activation)

                prediction = torch.argmax(output, dim=1).item()
                predictions.append(prediction)

    test_data = pd.DataFrame({'means': means, 'stds': stds, 'predictions': predictions,
                              'distributions': np.repeat(distributions, len(means) // len(distributions) + 1)[
                                               :len(means)]})
    return test_data


if __name__ == "__main__":
    start_prometheus_server()
    test_data = run_model()
    logger.info(f"Test data: \n{test_data.head()}")
    # Keep the application running to serve metrics to Prometheus
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus server")