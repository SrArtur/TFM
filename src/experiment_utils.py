import json
from datetime import datetime
from config import experiment_results_path
import logging

logger = logging.getLogger("TFM")


def save_experiment_results(experiment_name, experiment_metrics):
    """
    Saves the results of an experiment to a file in the results folder.
    :param experiment_name: The name of the experiment.
    :param experiment_metrics: The results of the experiment.
    :return: None
    """
    current_time = datetime.now()
    file_name = experiment_name + "_" + current_time.strftime("%H-%M_%d-%m-%Y") + ".json"
    file_path = experiment_results_path + file_name
    with open(file_path, "w") as json_file:
        json.dump(experiment_metrics, json_file, indent=4)

    logger.info(f"Metrics saved as {file_name} in {experiment_results_path[:-1]}")
    return file_path
