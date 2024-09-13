import torch
import logging
import prometheus_client
from prometheus_client import CollectorRegistry, Summary, push_to_gateway
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import MLP  # Importing the model class (MLP)
from data_loader import load_data  # Importing custom data loader
from statHook import StatHook  # For monitoring model statistics
import numpy as np
import time
from driftDetection import DriftDetectionPHT, DriftDetectionIKS, RelationshipTracker, NMWindow, Plover
from torchsummary import summary
from torchvision import models, transforms

# Setup logging to display and log information about the processes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_prometheus_server():
    """
    Starts the Prometheus server to expose metrics at port 8001.
    This is needed to monitor the model in real-time.
    """
    try:
        prometheus_client.start_http_server(8001)
        logger.info("Prometheus server started on port 8001")
    except Exception as e:
        logger.error(f"Error starting Prometheus server: {e}")

def prometheus_metrics():
    """
    Initializes and returns Prometheus metrics (Gauges) for tracking mean, std, and drift detection.
    """
    # Define Prometheus gauges to track model statistics and drift detection
    testing_mean_gauge = prometheus_client.Gauge('layer_output_testing_mean', 'Mean of the layer outputs')
    testing_std_gauge = prometheus_client.Gauge('layer_output_testing_std', 'Standard deviation of the layer outputs')
    
    # Drift detection metrics
    drift_pht_gauge = prometheus_client.Gauge('data_pht_drift_detection', 'PHT-based data drift detection (1 for drift, 0 for no drift)')
    drift_iks_gauge = prometheus_client.Gauge('data_iks_drift_detection', 'IKS-based data drift detection (1 for drift, 0 for no drift)')
    drift_nm_gauge = prometheus_client.Gauge('data_nm_drift_detection', 'NM-based data drift detection (1 for drift, 0 for no drift)')
    drift_plover_gauge = prometheus_client.Gauge('data_plover_drift_detection', 'Plover-based data drift detection (1 for drift, 0 for no drift)')
    
    return testing_mean_gauge, testing_std_gauge, drift_pht_gauge, drift_iks_gauge, drift_nm_gauge, drift_plover_gauge

def run_model():
    """
    Main function to load the model, dataset, and perform real-time drift detection.
    Metrics are exported to Prometheus for monitoring.
    """
    # Initialize Prometheus metrics for tracking
    testing_mean_gauge, testing_std_gauge, drift_pht_gauge, drift_iks_gauge, drift_nm_gauge, drift_plover_gauge = prometheus_metrics()
    
    # Initialize drift detection algorithms
    detector_PHT = DriftDetectionPHT()  # Page-Hinkley Test drift detection
    detector_IKS = DriftDetectionIKS(window_size=50)  # IKS drift detection
    class_survival_tracker = RelationshipTracker(50)  # Relationship tracking for features
    stat_drift_detector = NMWindow(window_size=50, threshold=0.1)  # NM-based drift detection
    detector_plover = Plover(window_size=50, threshold=0.1)  # Plover drift detection

    # Load the dataset and prepare training and test sets
    dataset = load_data('/home/elnaz/datasets/visionline')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    training_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Load the pre-trained model (MLP)
    model = MLP(3 * 128 * 128)
    model.load_state_dict(torch.load('model.pt', weights_only=True))  # Load the pre-trained weights
    model.eval()  # Set the model to evaluation mode

    stat_hook = StatHook()  # Initialize a hook to track the layer outputs

    ''' Training Phase: Update Reference Windows '''
    initial_means, initial_stds = [], []
    for images, labels_train in DataLoader(training_dataset, batch_size=1):
        with torch.no_grad():
            flat_images = images.view(images.size(0), -1)  # Flatten the input images
            output = model(flat_images)  # Get model output for each image
            stat_hook.hook_fn(model.fc2, flat_images, output)  # Track the outputs of the second fully connected layer
            if stat_hook.means and stat_hook.stds:
                initial_means.append(stat_hook.means[-1].item())
                initial_stds.append(stat_hook.stds[-1].item())
            detector_IKS.update_reference_windows(stat_hook.means[-1].clone().detach().unsqueeze(0))  # IKS reference window update
    
    # NM-DDM reference window update
    stat_drift_detector.update_reference_windows(initial_means, initial_stds)

    ''' Testing Phase: Perform Drift Detection '''
    test_loader = DataLoader(test_dataset, batch_size=1)
    drift_point = len(test_dataset) // 2  # Artificial drift point (optional)
    sum_drift = 0
    for i, (images_t, _) in enumerate(test_loader):
        with torch.no_grad():
            flat_images_t = images_t.view(images_t.size(0), -1)
            output = model(flat_images_t)
            stat_hook.hook_fn(model.fc2, flat_images_t, output)
            prediction = torch.argmax(output, dim=1).item()  # Get the predicted class
            
            if stat_hook.means and stat_hook.stds:
                current_mean = stat_hook.means[-1].item()
                current_std = stat_hook.stds[-1].item()
                testing_mean_gauge.set(current_mean)  # Update Prometheus gauge for mean
                testing_std_gauge.set(current_std)  # Update Prometheus gauge for std
                time.sleep(1)  # Simulate real-time data stream

                # Perform IKS-based drift detection
                has_drift, drift_iks_status = detector_IKS.update_detection_windows(stat_hook.means[-1].clone().detach().unsqueeze(0))
                drift_iks_gauge.set(1 if has_drift else 0)  # Update drift gauge for IKS

                # Perform PHT-based drift detection
                if sum_drift == 0 and detector_PHT.test_drift(current_mean, current_std):
                    logger.info("Drift detected using PHT!")
                    sum_drift += 1
                    drift_pht_gauge.set(1)
                else:
                    drift_pht_gauge.set(0)

                # Perform NM-DDM-based drift detection
                has_drift, drift_nm_status = stat_drift_detector.update_windows(current_mean, current_std)
                drift_nm_gauge.set(1 if has_drift else 0)  # Update drift gauge for NM-DDM

                # Perform Plover-based drift detection
                has_drift, drift_plover_status = detector_plover.update(current_mean, current_std)
                drift_plover_gauge.set(1 if has_drift else 0)  # Update drift gauge for Plover

if __name__ == "__main__":
    """
    Entry point to start the Prometheus server and run the model.
    Keeps the application running to serve real-time metrics to Prometheus.
    """
    start_prometheus_server()
    run_model()

    # Keep the application running indefinitely to serve metrics
    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down Prometheus server")
