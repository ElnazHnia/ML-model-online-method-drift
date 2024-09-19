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

This project demonstrates a machine learning pipeline for image classification and real-time drift detection using tools like Prometheus, Docker, and Evidently AI. The model monitors real-time performance and drift using several drift detection algorithms, ensuring that the deployed model maintains high accuracy in dynamic environments.

### Key features:
- Real-time monitoring of model metrics and data drift.
- Implementation of drift detection using **Page-Hinkley Test (PHT)**, **Incremental Kolmogorov-Smirnov (IKS)**, and **NM-DDM**.
- Visualization of metrics with **Prometheus** and **Grafana**.

## 2. Requirements

Ensure the following dependencies are installed before setting up the project:

- Python 3.x
- Docker (for containerization)
- Prometheus (for monitoring)
- Grafana (for visualization)
- PyTorch (for model training)
- Python libraries in `requirements.txt`.

## 3. Installation

To set up the project locally, follow these steps:

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
    If you want to containerize the application:
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
