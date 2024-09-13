import argparse
import logging
import train_model
import deployed_model

# Setup logging to capture and display information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to handle the mode of operation based on user input.
    The script can either train the model or deploy it with monitoring.
    """
    # Set up argument parsing to allow users to specify 'train' or 'deploy' modes
    parser = argparse.ArgumentParser(description='Train or Deploy Titanic Model')
    parser.add_argument('mode', choices=['train', 'deploy'], help='Mode to run the script in')
    args = parser.parse_args()

    # Handle the 'train' mode: start model training
    if args.mode == 'train':
        logger.info('Starting training...')
        train_model.train_model()  # Call the training function from train_model module

    # Handle the 'deploy' mode: start model deployment with monitoring
    elif args.mode == 'deploy':
        logger.info('Starting deployment...')
        deployed_model.start_prometheus_server()  # Start Prometheus server for monitoring
        test_data = deployed_model.run_model()  # Run the deployed model
        logger.info(f"Test data: \n{test_data.head()}")  # Log the test data (head of the DataFrame)

if __name__ == "__main__":
    main()
