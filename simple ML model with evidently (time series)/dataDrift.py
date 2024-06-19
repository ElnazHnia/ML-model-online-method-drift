import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# Creating a custom report
class DataDrift:
    def __init__(self):
        pass

    def generate_evidently_report(reference_data, current_data):
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html('data_drift_report.html')
        print("Evidently Data Drift Report is generated and saved as 'data_drift_report.html'.")
