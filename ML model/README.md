# ML Model Monitoring and Drift Detection

## Table of Contents
1. Introduction
2. Requirements
3. Installation
4. Repository Structure
5. Configuration
6. Running the Project
7. Monitoring and Drift Detection

## 1. Introduction

In many machine learning projects, effective monitoring is crucial for ensuring the quality, reliability, and sustainability of models in production. Poor monitoring can lead to significant challenges and pose risks. This article focuses on specific aspects of monitoring, such as performance metrics and drift detection. Monitoring these metrics is essential for validating and evaluating a model's performance during both development and deployment. Real-time data is collected on various aspects, allowing potential issues to be identified and adjustments to be made to maintain model quality and reliability. Available tools for monitoring performance metrics and detecting drift are also reviewed, helping organizations optimize and seamlessly deploy their machine learning applications. Continuous monitoring is key to ensuring the success and reliability of machine learning systems.

### Key features:
- Real-time monitoring of model metrics and data drift is supported.
- Drift detection is implemented using **Page-Hinkley Test (PHT)**, **Incremental Kolmogorov-Smirnov (IKS)**, and **NM-DDM**.
- Metrics are visualized with **Prometheus** and **Grafana**.

## 2. Requirements

The following dependencies should be installed before setting up the project:

- Python 3.x
- Docker (for containerization)
- Prometheus (for monitoring)
- Grafana (for visualization)
- PyTorch (for model training)
- Python libraries in `requirements.txt`.

## 3. Installation

To set up the project locally, the following steps should be followed:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd your-repo-name
    ```

3. **Set up a Python virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    ```

4. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Build Docker containers (optional)**:
    If Docker containers are needed, the following command can be run:
    ```bash
    docker-compose up --build
    ```

## 4. Repository Structure

```bash
├── configs/              # Configuration files
├── dataloader/           # Custom data loader
├── evaluation/           # Evaluation scripts
├── executor/             # Scripts to run the main model
├── model/                # Model definition and training
├── notebooks/            # Jupyter notebooks for experiments
├── ops/                  # Docker and deployment files
├── utils/                # Utility functions
└── README.md             # This file
```

## 5. Configuration

- **Prometheus**: The configuration for Prometheus alerts can be found in `configs/prometheus_rules.yaml`.
- **Docker**: The Docker setup is configured in the `docker-compose.yml` file located in the `ops/` folder.
- **Environment variables**: Ensure that environment variables are set properly. A `.env` file can be used in the project directory. If an example file is provided, it can be copied:
    ```bash
    cp .env.example .env
    ```

## 6. Running the Project

To run the project locally, follow these steps:

1. **Train the model**:
    The following command should be run to start the training process:
    ```bash
    python train_model.py
    ```

    The trained model will be saved as `model.pt` in the project directory.

2. **Start Prometheus server**:
    Ensure that Prometheus is installed and configured. The Prometheus server should be started by running:
    ```bash
    prometheus --config.file=path/to/prometheus.yml
    ```

3. **Run the deployed model**:
    After the model is trained and saved as `model.pt`, the deployed model with Prometheus monitoring enabled can be run:
    ```bash
    python deployed_model.py
    ```

4. **Run the deployed model using Docker**:
    If Docker is available, the deployed model can also be run in a container. Ensure Docker is configured properly and run:
    ```bash
    docker-compose up
    ```

## 7. Monitoring and Drift Detection

1. **Start Prometheus monitoring**:
    The deployed model script (`deployed_model.py`) will automatically start sending metrics to Prometheus, which will monitor model performance and detect drift.

2. **Drift detection**:
    Drift detection algorithms, such as **Page-Hinkley Test (PHT)**, **Incremental Kolmogorov-Smirnov (IKS)**, and **NM-DDM**, are implemented to detect data drift in real time. The drift status is visualized using **Prometheus** and can be monitored via **Grafana**.
