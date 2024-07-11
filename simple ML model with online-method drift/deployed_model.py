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
from driftDetection import DriftDetectionPHT, DriftDetectionIKS, RelationshipTracker, NMWindow, Plover
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


def prometheus_metrics():
    # Define and return Prometheus gauges
    testing_mean_gauge = prometheus_client.Gauge('layer_output_testing_mean', 'Mean of the layer outputs')
    testing_std_gauge = prometheus_client.Gauge('layer_output_testing_std',
                                                'Standard deviation of the layer outputs')
    drift_pht_gauge = prometheus_client.Gauge('data_pht_drift_detection',
                                              'Indicator of data drift detection (1 for drift, 0 for no drift)')
    drift_iks_gauge = prometheus_client.Gauge('data_iks_drift_detection',
                                              'Indicator of data drift detection (1 for drift, 0 for no drift)')
    drift_cd_gauge = prometheus_client.Gauge('data_cd_drift_detection',
                                             'Indicator of data drift detection (1 for drift, 0 for no drift)')
    drift_nm_gauge = prometheus_client.Gauge('data_nm_drift_detection',
                                             'Indicator of data drift detection (1 for drift, 0 for no drift)')
    drift_plover_gauge = prometheus_client.Gauge('data_plover_drift_detection',
                                                 'Indicator of data drift detection (1 for drift, 0 for no drift)')
    return testing_mean_gauge, testing_std_gauge, drift_pht_gauge, drift_iks_gauge, drift_cd_gauge, drift_nm_gauge, \
        drift_plover_gauge


def run_model():
    testing_mean_gauge, testing_std_gauge, drift_pht_gauge, drift_iks_gauge, drift_cd_gauge, \
        drift_nm_gauge, drift_plover_gauge = prometheus_metrics()
    detector_PHT = DriftDetectionPHT()
    features1 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    features_index = {feature: idx for idx, feature in enumerate(features1)}
    detector_IKS = DriftDetectionIKS(features1)
    class_survival_tracker = RelationshipTracker(50)
    stat_drift_detector = NMWindow(window_size=50, threshold=0.1)
    detector_plover = Plover(window_size=50, threshold=0.1)
    data, features = load_data()
    X_tensor, y_tensor = prepare_tensors(data, features)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    training_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    model = TitanicMLP(len(features))
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    stat_hook = StatHook()

    # IKS, CD-TDS, ref: NM-DDM
    # Update reference windows with training data
    '''' training Part '''
    initial_means, initial_stds = [], []
    for features_train, labels_train in DataLoader(training_dataset, batch_size=1):
        features_vector = features_train[0]
        feature_dict = {feature: features_vector[i].item() for i, feature in enumerate(features)}
        detector_IKS.update_reference_windows(feature_dict)
        # CD-TDS, Relationship among features for instance people in which Pclass survived
        pclass_data = features_train[:, features_index['Pclass']].item()
        survived_data = labels_train.item()  # Assuming batch size of 1 for simplicity
        class_survival_tracker.add_data(pclass_data, survived_data)
        has_cd_drift, drift_cd_status = class_survival_tracker.perform_drift_test()
        if has_cd_drift:
            drift_cd_gauge.set(1)
        elif not drift_cd_status:
            drift_cd_gauge.set(0)

        with torch.no_grad():
            output = model(features_train)
            stat_hook.hook_fn(model.fc2, features_vector, output)
            if stat_hook.means and stat_hook.stds:
                initial_means.append(stat_hook.means[-1].item())
                initial_stds.append(stat_hook.stds[-1].item())
    #  reference window : NM-DDM
    stat_drift_detector.update_reference_windows(initial_means, initial_stds)

    '''' testing Part '''
    test_loader = DataLoader(test_dataset, batch_size=1)
    # Artificial drift point
    drift_point = len(test_dataset) // 2
    sum_drift = 0
    for i, (features_t, _) in enumerate(test_loader):
        with torch.no_grad():
            single_input = features_t.unsqueeze(0)

            # Artificial drift introduction
            if i >= drift_point:
                features_t += torch.randn_like(features_t) * 200  # Injecting noise to create drift

            # IKS - compare distribution between 2 windows for all features
            # limitation: waited until the detection window will be filled.
            # Update detection windows with testing data
            features_vector1 = features_t[0]
            feature_dict = {feature: features_vector1[i].item() for i, feature in enumerate(features)}
            has_drift, drift_iks_status = detector_IKS.update_detection_windows(feature_dict)
            if has_drift:
                drift_iks_gauge.set(1)
            elif not drift_iks_status:
                drift_iks_gauge.set(0)
            output = model(features_t)
            prediction = torch.argmax(output, dim=1).item()
            stat_hook.hook_fn(model.fc2, single_input, output)
            if stat_hook.means and stat_hook.stds:
                current_mean = stat_hook.means[-1].item()
                current_std = stat_hook.stds[-1].item()
                testing_mean_gauge.set(current_mean)
                testing_std_gauge.set(current_std)
                time.sleep(1)
                # PHT - use just 1 sequence
                if sum_drift == 0:
                    if detector_PHT.test_drift(current_mean, current_std):
                        logger.info("Drift detected!")
                        sum_drift += 1
                        drift_pht_gauge.set(1)

                    else:
                        # dynamic_window.update_reference_data(current_mean, current_std)
                        drift_pht_gauge.set(0)
                        # logger.info("No drift detected. Model is stable.")
                # NM-DDM, probability density functions (PDFs) for calculating the log-likelihood ratios of means and
                # standard deviations
                has_drift, drift_nm_status = stat_drift_detector.update_windows(current_mean, current_std)
                if has_drift:
                    logger.info("In NM-DDM Drift detected!")
                    drift_nm_gauge.set(1)
                elif not drift_nm_status:
                    drift_nm_gauge.set(0)
                # Plover,
                has_drift, drift_plover_status = detector_plover.update(current_mean, current_std)
                if has_drift:
                    logger.info("In Plover Drift detected!")
                    drift_plover_gauge.set(1)
                elif not drift_plover_status:
                    drift_plover_gauge.set(0)


if __name__ == "__main__":
    start_prometheus_server()
    run_model()
    # time.sleep(60)
    # Keep the application running to serve metrics to Prometheus
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus server")
