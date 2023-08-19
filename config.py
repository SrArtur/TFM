import logging

# Logging configuration
logFormatter = logging.Formatter('%(name)s - [%(levelname)s] - %(asctime)s - %(funcName)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(level=logging.INFO, format='%(name)s - [%(levelname)s] - %(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
rootLogger = logging.getLogger("TFM")
rootLogger.setLevel(logging.WARNING)

fileHandler = logging.FileHandler('logs/main.log')  # For logging to file
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

# Paths configuration
adult_data_path = "data/adult.data"
breast_cancer_data_path = "data/breast-cancer.data"
data_path = "data/"

experiment_results_path = "logs/experiment_log/"
