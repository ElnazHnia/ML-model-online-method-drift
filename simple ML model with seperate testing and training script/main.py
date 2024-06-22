import argparse
import logging
import train_model
import deployed_model
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train or Deploy Titanic Model')
    parser.add_argument('mode', choices=['train', 'deploy'], help='Mode to run the script in')
    args = parser.parse_args()

    if args.mode == 'train':
        logger.info('Starting training...')
        train_model.train_model()
    elif args.mode == 'deploy':
        logger.info('Starting deployment...')
        deployed_model.start_prometheus_server()
        test_data = deployed_model.run_model()
        logger.info(f"Test data: \n{test_data.head()}")


if __name__ == "__main__":
    main()
