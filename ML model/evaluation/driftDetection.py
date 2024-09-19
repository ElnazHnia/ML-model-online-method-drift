from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde

# IKS - Incremental Kolmogorov-Smirnov Drift Detection
class FeatureWindow:
    """
    A class for maintaining reference and detection windows to perform drift detection 
    using the Kolmogorov-Smirnov (KS) test.
    """
    def __init__(self, window_size=50):
        self.reference_window = deque(maxlen=window_size)
        self.detection_window = deque(maxlen=window_size)
        self.window_size = window_size

    def add_to_detection_window(self, value):
        """
        Add values to the detection window and perform the KS drift test if the window is full.
        """
        self.detection_window.append(value.cpu().numpy().flatten())
        if len(self.detection_window) >= self.window_size:
            return self.perform_drift_test()
        return False  # Return False if no test was performed

    def add_to_reference_window(self, value):
        """
        Add values to the reference window.
        """
        self.reference_window.append(value.cpu().numpy().flatten())

    def perform_drift_test(self):
        """
        Perform the KS drift test between the reference and detection windows.
        """
        if len(self.reference_window) >= self.reference_window.maxlen:
            ref_data = np.array(self.reference_window).reshape(self.window_size, -1)
            det_data = np.array(self.detection_window).reshape(self.window_size, -1)
            _, p_value = ks_2samp(ref_data.flatten(), det_data.flatten())

            drift_detected = p_value < 0.05
            self.detection_window.clear()  # Optionally reset detection window after the test
            return drift_detected
        return False  # Return False if insufficient data for a reliable test


class DriftDetectionIKS:
    """
    A class that uses IKS-based drift detection.
    """
    def __init__(self, window_size=50):
        self.feature_windows = FeatureWindow(window_size)

    def update_detection_windows(self, feature_tensor):
        """
        Update detection windows with new feature data and check for drift.
        """
        drift_detected = self.feature_windows.add_to_detection_window(feature_tensor)
        if drift_detected:
            print("Drift detected using IKS!")
            return drift_detected, {'value': 'has drift'}
        else:
            return False, {}

    def update_reference_windows(self, feature_tensor):
        """
        Update reference windows with initial feature data.
        """
        self.feature_windows.add_to_reference_window(feature_tensor)


# CD-TDS - Change Detection in Transactional Data Stream
class RelationshipTracker:
    """
    A class to track relationships between two data streams and detect drift using 
    a Chi-square test.
    """
    def __init__(self, window_size=50):
        self.data_x = deque(maxlen=window_size)
        self.data_y = deque(maxlen=window_size)

    def add_data(self, value_x, value_y):
        """
        Add data to the respective windows for X and Y.
        """
        self.data_x.append(value_x)
        self.data_y.append(value_y)

    def perform_drift_test(self):
        """
        Perform the Chi-square test to detect drift between data streams.
        """
        if len(self.data_x) == self.data_x.maxlen:
            contingency_table = pd.crosstab(pd.Series(self.data_x), pd.Series(self.data_y))
            _, p_value, _, _ = chi2_contingency(contingency_table)
            return p_value < 0.05, {'value': p_value}  # Drift detected if p-value is low
        return False, {}


# PHT - Page-Hinkley Test for drift detection
class PageHinkley:
    """
    A class for detecting concept drift using the Page-Hinkley Test (PHT), 
    designed to identify abrupt changes in data streams.
    """
    def __init__(self, delta=0.005, lambda_=50, alpha=1.0):
        self.delta = delta  # Detection threshold
        self.lambda_ = lambda_  # Minimum observations required before detecting drift
        self.alpha = alpha  # Allowable magnitude of change
        self.cumulative_sum = 0
        self.avg = 0
        self.observation_count = 0
        self.min_cumulative_sum = 0

    def update(self, value):
        """
        Update the cumulative sum with a new observation and check for drift.
        """
        if self.observation_count > 0:
            self.cumulative_sum += value - self.avg - self.delta
        else:
            self.cumulative_sum = 0
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        self.avg = ((self.avg * self.observation_count) + value) / (self.observation_count + 1)
        self.observation_count += 1

        return self.cumulative_sum - self.min_cumulative_sum > self.lambda_


class DriftDetectionPHT:
    """
    A class for detecting drift using the Page-Hinkley Test for both mean and standard deviation.
    """
    def __init__(self, delta=0.005, lambda_=50, alpha=1.0):
        self.ph_mean = PageHinkley(delta, lambda_, alpha)
        self.ph_std = PageHinkley(delta, lambda_, alpha)

    def test_drift(self, current_mean, current_std):
        """
        Test for drift based on changes in mean and standard deviation.
        """
        mean_drift = self.ph_mean.update(current_mean)
        std_drift = self.ph_std.update(current_std)
        return mean_drift or std_drift


# NM-DDM - Nonparametric Multidimensional Drift Detection Method
class NMWindow:
    """
    A class for detecting drift using nonparametric methods and log-likelihood ratios
    between two windows (reference and detection) for mean and standard deviation.
    """
    def __init__(self, window_size=50, threshold=0.1):
        self.mean_reference_NM_window = deque(maxlen=window_size)
        self.std_reference_NM_window = deque(maxlen=window_size)
        self.mean_detection_NM_window = deque(maxlen=window_size)
        self.std_detection_NM_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.drift_nm_status = {}

    def update_windows(self, current_mean, current_std):
        """
        Update the detection windows and test for drift if enough data is available.
        """
        self.mean_detection_NM_window.append(current_mean)
        self.std_detection_NM_window.append(current_std)

        if len(self.mean_detection_NM_window) >= self.mean_detection_NM_window.maxlen:
            has_drift = self.perform_drift_test()
            if has_drift:
                self.drift_nm_status = {'value': 'has drift'}
                return True, self.drift_nm_status

        return False, self.drift_nm_status

    def estimate_pdf(self, data):
        """
        Estimate the probability density function using Kernel Density Estimation (KDE).
        """
        if len(data) > 1:
            kde = gaussian_kde(data)
            return kde
        return None

    def update_reference_windows(self, means, stds):
        """
        Update the reference windows with the provided means and standard deviations.
        """
        self.mean_reference_NM_window.extend(means)
        self.std_reference_NM_window.extend(stds)

    def perform_drift_test(self):
        """
        Perform drift detection using log-likelihood ratios for the reference and detection windows.
        """
        mean_ref_pdf = self.estimate_pdf(np.array(self.mean_reference_NM_window))
        mean_det_pdf = self.estimate_pdf(np.array(self.mean_detection_NM_window))
        std_ref_pdf = self.estimate_pdf(np.array(self.std_reference_NM_window))
        std_det_pdf = self.estimate_pdf(np.array(self.std_detection_NM_window))

        if mean_ref_pdf is not None and mean_det_pdf is not None and std_ref_pdf is not None and std_det_pdf is not None:

            mean_ll_ratios = np.zeros(len(self.mean_detection_NM_window))
            std_ll_ratios = np.zeros(len(self.std_detection_NM_window))

            for i in range(len(self.mean_detection_NM_window)):
                mean_ll_ratios[i] = np.log(mean_det_pdf(self.mean_detection_NM_window[i]) / mean_ref_pdf(self.mean_detection_NM_window[i])) if mean_ref_pdf(self.mean_detection_NM_window[i]) and mean_det_pdf(self.mean_detection_NM_window[i]) else 0
                std_ll_ratios[i] = np.log(std_det_pdf(self.std_detection_NM_window[i]) / std_ref_pdf(self.std_detection_NM_window[i])) if std_ref_pdf(self.std_detection_NM_window[i]) and std_det_pdf(self.std_detection_NM_window[i]) else 0

            max_mean_ll_ratio = max(mean_ll_ratios)
            max_std_ll_ratio = max(std_ll_ratios)

            # Clear detection windows after test
            self.mean_detection_NM_window.clear()
            self.std_detection_NM_window.clear()

            # Check if any of the ratios exceed the threshold
            if max_mean_ll_ratio > self.threshold or max_std_ll_ratio > self.threshold:
                self.mean_reference_NM_window.extend(self.mean_detection_NM_window)
                self.std_reference_NM_window.extend(self.std_detection_NM_window)
                return True
            
        return False


# Plover - Proof-of-concept Drift Detection
class Plover:
    """
    A proof-of-concept class for detecting drift by monitoring changes in mean and standard deviation 
    between reference and detection windows.
    """
    def __init__(self, window_size=50, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window_mean = []
        self.detection_window_mean = []
        self.reference_window_std = []
        self.detection_window_std = []
        self.initialized = False
        self.drift_plover_status = {}

    def initialize(self, initial_data_mean, initial_data_std):
        """
        Initialize the reference window with initial data.
        """
        if len(initial_data_mean) >= self.window_size and len(initial_data_std) >= self.window_size:
            self.reference_window_mean = initial_data_mean[:self.window_size]
            self.reference_window_std = initial_data_std[:self.window_size]
            self.initialized = True
        else:
            raise ValueError("Initial data length must be at least equal to window size for both mean and std.")

    def update(self, current_mean, current_std):
        """
        Update detection window with new data and check for drift based on divergence.
        """
        if not self.initialized:
            self.initialize([current_mean] * self.window_size, [current_std] * self.window_size)
            return False, self.drift_plover_status

        self.detection_window_mean.append(current_mean)
        self.detection_window_std.append(current_std)

        if len(self.detection_window_mean) > self.window_size:
            self.detection_window_mean.pop(0)
        if len(self.detection_window_std) > self.window_size:
            self.detection_window_std.pop(0)

        if len(self.detection_window_mean) < self.window_size or len(self.detection_window_std) < self.window_size:
            return False, self.drift_plover_status

        # Compute statistics for reference and detection windows
        reference_mean = np.mean(self.reference_window_mean)
        reference_std = np.mean(self.reference_window_std)

        detection_mean = np.mean(self.detection_window_mean)
        detection_std = np.mean(self.detection_window_std)

        # Calculate divergence (absolute difference between means and stds)
        divergence_mean = np.abs(reference_mean - detection_mean)
        divergence_std = np.abs(reference_std - detection_std)

        # Update reference window periodically
        if len(self.detection_window_mean) % self.window_size == 0:
            self.reference_window_mean = self.detection_window_mean[:]
            self.reference_window_std = self.detection_window_std[:]

        # Check for drift based on the threshold for both mean and std
        if divergence_mean > self.threshold or divergence_std > self.threshold:
            self.drift_plover_status = {'value': 'has drift'}
            return True, self.drift_plover_status  # Drift detected
        else:
            return False, self.drift_plover_status  # No drift detected
