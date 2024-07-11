from collections import deque
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde


# IKS
class FeatureWindow:
    def __init__(self, window_size=50):
        self.reference_window = deque(maxlen=window_size)
        self.detection_window = deque(maxlen=window_size)
        self.window_size = window_size

    def add_to_detection_window(self, value):
        self.detection_window.append(value)
        if len(self.detection_window) >= self.window_size:
            return self.perform_drift_test()
        return False  # Return False if no test was performed

    def add_to_reference_window(self, value):
        self.reference_window.append(value)

    def perform_drift_test(self):
        if len(self.reference_window) >= self.reference_window.maxlen:
            _, p_value = ks_2samp(list(self.reference_window), list(self.detection_window))

            drift_detected = p_value < 0.05
            # Optionally reset detection window after a test
            self.detection_window.clear()
            return drift_detected
        return False  # Return False if there's insufficient data for a reliable test


class DriftDetectionIKS:
    def __init__(self, features, window_size=50):
        self.feature_windows = {feature: FeatureWindow(window_size) for feature in features}
        self.drift_status = {}

    def update_detection_windows(self, feature_dict):
        drift_detected = False
        for feature, value in feature_dict.items():
            if self.feature_windows[feature].add_to_detection_window(value):
                drift_detected = True
                self.drift_status = {'value': value}
                print(f"Drift detected in feature: {feature}")
        return drift_detected, self.drift_status

    def update_reference_windows(self, feature_dict):
        for feature, value in feature_dict.items():
            self.feature_windows[feature].add_to_reference_window(value)


# CD-TDS
class RelationshipTracker:
    def __init__(self, window_size=50):
        self.data_x = deque(maxlen=window_size)
        self.data_y = deque(maxlen=window_size)

    def add_data(self, value_x, value_y):
        self.data_x.append(value_x)
        self.data_y.append(value_y)

    def perform_drift_test(self):
        if len(self.data_x) == self.data_x.maxlen:
            # Example test: Chi-squared test for independence in categorical data
            contingency_table = pd.crosstab(pd.Series(self.data_x), pd.Series(self.data_y))
            _, p_value, _, _ = chi2_contingency(contingency_table)
            return p_value < 0.05, {'value': p_value}  # Drift detected if p-value is low
        return False, {}


# PHT
class PageHinkley:
    def __init__(self, delta=0.005, lambda_=50, alpha=1.0):
        self.delta = delta  # Detection threshold
        self.lambda_ = lambda_  # Minimum number of observations before detecting change
        self.alpha = alpha  # Magnitude of allowable change
        self.cumulative_sum = 0
        self.avg = 0
        self.observation_count = 0
        self.min_cumulative_sum = 0

    def update(self, value):

        if self.observation_count > 0:
            self.cumulative_sum += value - self.avg - self.delta
        else:
            self.cumulative_sum = 0
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        self.avg = ((self.avg * self.observation_count) + value) / (self.observation_count + 1)
        self.observation_count += 1

        return self.cumulative_sum - self.min_cumulative_sum > self.lambda_


class DriftDetectionPHT:
    def __init__(self, delta=0.005, lambda_=50, alpha=1.0):
        self.ph_mean = PageHinkley(delta, lambda_, alpha)
        self.ph_std = PageHinkley(delta, lambda_, alpha)

    def test_drift(self, current_mean, current_std):
        mean_drift = self.ph_mean.update(current_mean)
        std_drift = self.ph_std.update(current_std)
        return mean_drift or std_drift


# NM-DDM
class NMWindow:

    def __init__(self, window_size=50, threshold=0.1):

        self.mean_reference_NM_window = deque(maxlen=window_size)
        self.std_reference_NM_window = deque(maxlen=window_size)
        self.mean_detection_NM_window = deque(maxlen=window_size)
        self.std_detection_NM_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.drift_nm_status = {}

    def update_windows(self, current_mean, current_std):
        self.mean_detection_NM_window.append(current_mean)
        self.std_detection_NM_window.append(current_std)

        if len(self.mean_detection_NM_window) >= self.mean_detection_NM_window.maxlen:
            has_drift = self.perform_drift_test()
            if has_drift:
                self.drift_nm_status = {'value': 'has drift'}
                return True, self.drift_nm_status

        return False, self.drift_nm_status

    def estimate_pdf(self, data):
        return gaussian_kde(data) if len(data) > 1 else None

    def update_reference_windows(self, means, stds):
        self.mean_reference_NM_window.extend(means)
        self.std_reference_NM_window.extend(stds)

    def perform_drift_test(self):

        mean_ref_pdf = self.estimate_pdf(np.array(self.mean_reference_NM_window))
        mean_det_pdf = self.estimate_pdf(np.array(self.mean_detection_NM_window))
        std_ref_pdf = self.estimate_pdf(np.array(self.std_reference_NM_window))
        std_det_pdf = self.estimate_pdf(np.array(self.std_detection_NM_window))

        if mean_ref_pdf is not None and mean_det_pdf is not None and std_ref_pdf is not None and std_det_pdf is not None:

            mean_ll_ratios = [np.log(mean_det_pdf(value) / mean_ref_pdf(value)) for value in
                              self.mean_detection_NM_window]
            std_ll_ratios = [np.log(std_det_pdf(value) / std_ref_pdf(value)) for value in self.std_detection_NM_window]

            max_mean_ll_ratio = max(mean_ll_ratios)
            max_std_ll_ratio = max(std_ll_ratios)

            # Clear detection windows after test
            self.mean_detection_NM_window.clear()
            self.std_detection_NM_window.clear()

            # Check if any of the ratios exceed the threshold
            if max_mean_ll_ratio > self.threshold or max_std_ll_ratio > self.threshold:
                # Update reference windows to the latest state before the drift
                self.mean_reference_NM_window = deque(self.mean_detection_NM_window,
                                                      maxlen=self.mean_reference_NM_window.maxlen)
                self.std_reference_NM_window = deque(self.std_detection_NM_window,
                                                     maxlen=self.std_reference_NM_window.maxlen)
                return True
        return False


class Plover:
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
        if len(initial_data_mean) >= self.window_size and len(initial_data_std) >= self.window_size:
            self.reference_window_mean = initial_data_mean[:self.window_size]
            self.reference_window_std = initial_data_std[:self.window_size]
            self.initialized = True
        else:
            raise ValueError("Initial data length must be at least equal to window size for both mean and std.")

    def update(self, current_mean, current_std):
        if not self.initialized:
            self.initialize([current_mean] * self.window_size, [current_std] * self.window_size)
            return False, self.drift_plover_status

        self.detection_window_mean.append(current_mean)
        self.detection_window_std.append(current_std)

        if len(self.detection_window_mean) > self.window_size:
            self.detection_window_mean.pop(0)
        if len(self.detection_window_std) > self.window_size:
            self.detection_window_std.pop(0)

        # Check if enough data points in detection window
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

        # Check for drift based on threshold for both mean and std
        if divergence_mean > self.threshold or divergence_std > self.threshold:
            self.drift_plover_status = {'value': 'has drift'}
            return True, self.drift_plover_status  # Drift detected
        else:
            return False, self.drift_plover_status  # No drift detected
