ML Model Monitoring and Drift Detection
Table of Contents

    1. Introduction
    2. Requirements
    3. Installation
    4. Repository Structure
    5. Configuration
    6. Running the Project
    7. Monitoring and Drift Detection
    

1. Introduction

This project demonstrates a machine learning pipeline for image classification and real-time drift detection using tools like Prometheus, Docker, and Evidently AI. The model monitors real-time performance and drift using several drift detection algorithms, ensuring that the deployed model maintains high accuracy in dynamic environments.

Key features:

    Real-time monitoring of model metrics and data drift.
    Implementation of drift detection using Page-Hinkley Test (PHT), Incremental Kolmogorov-Smirnov (IKS), and NM-DDM.
    Visualization of metrics with Prometheus and Grafana.

2. Requirements

Ensure the following dependencies are installed before setting up the project:

    Python 3.x
    Docker (for containerization)
    Prometheus (for monitoring)
    Grafana (for visualization)
    PyTorch (for model training)
    Python libraries in requirements.txt.

3. Installation

To set up the project locally, follow these steps:

    Clone the repository:

    bash

git clone https://github.com/your-username/your-repo-name.git

Navigate to the project directory:

bash

cd your-repo-name

Set up a Python virtual environment (optional but recommended):

bash

python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

Install required dependencies:

bash

pip install -r requirements.txt

Build Docker containers (optional): If you want to containerize the application:

bash

    docker-compose up --build

4. Repository Structure

bash

├── configs/              # Configuration files
├── dataloader/           # Custom data loader
├── evaluation/           # Evaluation scripts
├── executor/             # Scripts to run the main model
├── model/                # Model definition and training
├── notebooks/            # Jupyter notebooks for experiments
├── ops/                  # Docker and deployment files
├── utils/                # Utility functions
└── README.md             # This file

5. Configuration

    Prometheus: Configuration files for Prometheus alerts are located in configs/prometheus_rules.yaml.
    Docker: The Docker setup can be configured in docker-compose.yml inside the ops/ folder.
    Environment variables: You may need to set environment variables in a .env file. Check for example .env.example.

6. Running the Project

To run the project:

    Run the model locally:

    bash

python train_model.py

Deploy the model and monitor with Prometheus: Start the Prometheus server:

bash

prometheus --config.file=path/to/prometheus.yml

For Docker-based deployment:

bash

    docker-compose up

7. Monitoring and Drift Detection

    Start Prometheus server: Prometheus will begin collecting metrics by running:

    bash

python deployed_model.py

Drift detection: The drift detection is implemented using Page-Hinkley Test (PHT), IKS, and NM-DDM algorithms. Real-time drift status is updated in Prometheus and visualized in Grafana.