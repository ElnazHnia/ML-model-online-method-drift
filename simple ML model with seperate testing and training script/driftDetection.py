import torch
import logging
import pandas as pd
import prometheus_client
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import TitanicMLP
from data_loader import load_data, prepare_tensors
from statHook import StatHook
from datetime import datetime, timedelta
from scipy.stats import ks_2samp, chisquare
import numpy as np
import time
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetection:
    def __init__(self):

        self.training_mean_gauge, self.training_std_gauge, self.drift_p_value_gauge, self.drift_detected_gauge, \
            self.drift_test_gauge, self.reference_data_mean_gauge, self.current_data_mean_gauge, \
            self.reference_data_std_gauge, self.current_data_std_gauge,\
            self.survival_predictions_gauge, self.drift_concept_detected_gauge,\
            self.drift_Pclass_gauge, self.drift_Sex_gauge,\
            self.drift_Age_gauge, self.drift_SibSp_gauge, self.drift_Parch_gauge, self.drift_Fare_gauge,\
            self.drift_Embarked_gauge = self.prometheus_metrics()

    def prometheus_metrics(self):

        training_mean_gauge = prometheus_client.Gauge('layer_output_train_mean', 'Mean of the layer outputs')
        training_std_gauge = prometheus_client.Gauge('layer_output_train_std',
                                                     'Standard deviation of the layer outputs')
        drift_test_gauge = prometheus_client.Gauge('data_drift_test_result', 'Result of data drift test')
        drift_p_value_gauge = prometheus_client.Gauge('data_drift_p_value', 'P-value for data drift detection')
        drift_detected_gauge = prometheus_client.Gauge('data_drift_detected',
                                                       'Data drift detected (1 if detected, 0 otherwise)')

        reference_data_mean_gauge = prometheus_client.Gauge('reference_data_mean',
                                                            'Reference data mean for drift detection')
        current_data_mean_gauge = prometheus_client.Gauge('current_data_mean', 'Current data mean for drift detection')
        reference_data_std_gauge = prometheus_client.Gauge('reference_data_std',
                                                           'Reference data std for drift detection')
        current_data_std_gauge = prometheus_client.Gauge('current_data_std', 'Current data std for drift detection')
        survival_predictions_gauge = prometheus_client.Gauge('survival_predictions_gauge',
                                                              'Reference data std for drift detection', ['survived'])
        drift_concept_detected_gauge = prometheus_client.Gauge('drift_concept_detected', 'Drift of the target')
        drift_Pclass_gauge = prometheus_client.Gauge('drift_Pclass_gauge', 'Drift of the feature')
        drift_Sex_gauge = prometheus_client.Gauge('drift_Sex_gauge', 'Drift of the feature')
        drift_Age_gauge = prometheus_client.Gauge('drift_Age_gauge', 'Drift of the feature')
        drift_SibSp_gauge = prometheus_client.Gauge('drift_SibSp_gauge', 'Drift of the feature')
        drift_Parch_gauge = prometheus_client.Gauge('drift_Parch_gauge', 'Drift of the feature')
        drift_Fare_gauge = prometheus_client.Gauge('drift_Fare_gauge', 'Drift of the feature')
        drift_Embarked_gauge = prometheus_client.Gauge('drift_Embarked_gauge', 'Drift of the feature')
        return training_mean_gauge, training_std_gauge, drift_p_value_gauge, drift_detected_gauge, drift_test_gauge, \
            reference_data_mean_gauge, current_data_mean_gauge, reference_data_std_gauge, current_data_std_gauge, \
            survival_predictions_gauge,drift_concept_detected_gauge, drift_Pclass_gauge, drift_Sex_gauge,\
            drift_Age_gauge, drift_SibSp_gauge, drift_Parch_gauge, drift_Fare_gauge, drift_Embarked_gauge




    def detect_ks_2samp_drift(self, reference_data, current_data):
        '''
           model drift detection (example using Kolmogorov-Smirnov test for numerical features)
        '''
        stat, p_value = ks_2samp(reference_data, current_data)
        drift_detected = p_value < 0.05
        self.drift_p_value_gauge.set(p_value)
        self.drift_detected_gauge.set(1 if drift_detected else 0)
        logging.info(f"Data drift p-value: {p_value}, Drift detected: {drift_detected}")
        return drift_detected

    def detect_chisquare_drift(self, reference_data, current_data):
        '''
           model drift detection (example using Kolmogorov-Smirnov test for categorical features)
        '''
        stat, p_value = chisquare(reference_data, current_data)
        drift_detected = p_value < 0.05
        self.drift_p_value_gauge.set(p_value)
        self.drift_detected_gauge.set(1 if drift_detected else 0)
        logging.info(f"Data drift p-value: {p_value}, Drift detected: {drift_detected}")
        return drift_detected

    def detect_model_Mean_curr_drift(self, means):
        # Mean

        current_mean_data = np.array(means[int(0.8 * len(means)):])
        for value in current_mean_data:
            self.current_data_mean_gauge.set(value)
            logger.info(f"current_mean_data: {value}")
            time.sleep(10)


    def detect_model_Mean_ref_drift(self, means):
        # Mean
        reference_mean_data = np.array(means[:int(0.8 * len(means))])

        for value in reference_mean_data:
            self.reference_data_mean_gauge.set(value)
            logger.info(f"reference_mean_data: {value}")
            time.sleep(10)


    def detect_model_STD_curr_drift(self, stds):
        # STD

        current_std_data = np.array(stds[int(0.8 * len(stds)):])
        for value in current_std_data:
            self.current_data_std_gauge.set(value)
            logger.info(f"current_std_data: {value}")
            time.sleep(10)

    def detect_model_STD_ref_drift(self, stds):
        # STD
        reference_std_data = np.array(stds[:int(0.8 * len(stds))])

        for value in reference_std_data:
            self.reference_data_std_gauge.set(value)
            logger.info(f"reference_std_data: {value}")
            time.sleep(10)

    def detect_concept_drift(self, predictions):
        # Placeholder for concept drift detection logic
        # predictions
        survived_count = sum(predictions)
        not_survived_count = len(predictions) - survived_count

        self.survival_predictions_gauge.labels(survived='1').set(survived_count)
        self.survival_predictions_gauge.labels(survived='0').set(not_survived_count)
        # Assume you have drift detection logic here
        drift_detected = survived_count / (survived_count + not_survived_count) > 0.5
        self.drift_concept_detected_gauge.set(1 if drift_detected else 0)
        logger.info(f"Concept drift detected: {drift_detected}")
    def detect_data_drift(self, reference_data, current_data):
        """
                Detect data drift between reference and current data using statistical tests.
                - For numerical features, use the Kolmogorov-Smirnov test.
                - For categorical features, use the Chi-square test.
         """
        drift_detected = False
        for column in reference_data.columns:
            feature_name = f"drift_{column}_gauge"
            logging.info(f"Looking for feature name: {feature_name}")
            gauge = getattr(self, feature_name, None)
            if gauge is None:
                logging.warning(f"Gauge for column {column} does not exist.")
            else:
                logging.info(f"Gauge {gauge}")

            if pd.api.types.is_numeric_dtype(reference_data[column]):
                stat, p_value = ks_2samp(reference_data[column], current_data[column])

            else:
                stat, p_value = chisquare(reference_data[column].value_counts(), current_data[column].value_counts())

            if p_value < 0.05:
                drift_detected = True
                self.drift_p_value_gauge.set(p_value)
                gauge.set(1)
                logging.info(f"Data drift detected in column {column} with p-value: {p_value}")
                # break  # Stop checking further if drift is detected in any column
            else:
                logging.info(f"Data drift detected in column {column} with p-value: {p_value}")
                gauge.set(0)
        if not drift_detected:
            # self.drift_detected_gauge.set(0)
            logging.info("No data drift detected.")

    def evidently_drift_tests(self, test_data):
        ''' EvidentlyAI data drift tests  '''
        logger.info('EvidentlyAI data drift tests')
        per_column_stattest = {x: 'wasserstein' for x in ['means', 'stds']}
        data_drift_dataset_tests = TestSuite(tests=[
            TestShareOfDriftedColumns(per_column_stattest=per_column_stattest),
        ])
        data_drift_dataset_tests.run(reference_data=test_data[:int(0.8 * len(test_data))],
                                     current_data=test_data[int(0.8 * len(test_data)):])
        drift_result = data_drift_dataset_tests.as_dict()
        drift_detected_count = sum(1 for test in drift_result['tests'] if test['status'] == 'FAIL')
        self.drift_test_gauge.set(drift_detected_count)
        logger.info(f"Drift result: {drift_result}")

    def run_drift_detection(self, means, stds, predictions, test_data, reference_data, current_data):
        model_drift_mean_ref_thread = threading.Thread(target=self.detect_model_Mean_ref_drift, args=(means,))
        model_drift_mean_curr_thread = threading.Thread(target=self.detect_model_Mean_curr_drift, args=(means,))
        model_drift_std_curr_thread = threading.Thread(target=self.detect_model_STD_curr_drift, args=(stds,))
        model_drift_std_ref_thread = threading.Thread(target=self.detect_model_STD_ref_drift, args=(stds,))
        concept_drift_thread = threading.Thread(target=self.detect_concept_drift, args=(predictions,))
        evidently_drift_thread = threading.Thread(target=self.evidently_drift_tests, args=(test_data,))
        detect_data_drift_thread = threading.Thread(target=self.detect_data_drift, args=(reference_data, current_data,))

        detect_data_drift_thread.start()
        evidently_drift_thread.start()
        model_drift_mean_ref_thread.start()
        model_drift_mean_curr_thread.start()
        model_drift_std_curr_thread.start()
        model_drift_std_ref_thread.start()
        concept_drift_thread.start()

        detect_data_drift_thread.join()
        evidently_drift_thread.join()
        model_drift_mean_ref_thread.join()
        model_drift_mean_curr_thread.join()
        model_drift_std_curr_thread.join()
        model_drift_std_ref_thread.join()
        concept_drift_thread.join()


