import argparse
import json
import logging
import os
import random
import time

import numpy as np

from src.datasets import preprocess_data, \
    load_client_train_test_split
from src.experiment_utils import save_experiment_results
from src.models import load_simple_model_with_more_params, load_ensemble, set_base_learner, \
    load_simple_model, evaluate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger("TFM")
logger.setLevel(logging.INFO)
global exp_name


def execute_experiment(num_parties: int, data_preproc: np.ndarray, data_labels: np.ndarray, model_type: str = "simple",
                       rounds: int = 2):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    models = []
    metrics = {}

    for i in range(1, num_parties + 1):  # Por el id
        train, test, train_label, test_label = load_client_train_test_split(num_parties=num_parties, client_id=i,
                                                                            data=data_preproc, labels=data_labels)
        train_data.append(train)
        test_data.append(test)
        train_labels.append(train_label)
        test_labels.append(test_label)
        if model_type == "simple":
            model = load_simple_model(train.shape[1], f'model{i - 1}')
        elif model_type == "complex":
            model = load_simple_model_with_more_params(train.shape[1], f'model{i - 1}')
        else:
            raise ValueError(f"Model type {model_type} not supported")
        models.append(model)

        logger.info(f"Splitting data for client {i} with  train shape {train.shape} and test shape {test.shape}")

    ensemble = load_ensemble(models)
    ensemble.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    # rounds = 2

    for r in range(0, rounds):

        metrics[r] = {}
        centralized_metrics = {}  # Metrics when ensemble has been trained
        decentralized_metrics = {}  # Metrics when models have been trained
        round_metrics = {}

        logger.info(f"# Round {r} #")

        # Entrenamiento y evaluación descentralizada
        for i in range(num_parties):
            model = models[i]
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

            logger.info(f"Training model{i} - round {r}")
            model.fit(train_data[i], train_labels[i], epochs=4, batch_size=128, verbose=0)
            loss, acc, mse, auc, f1 = evaluate_model(model, test_data[i], test_labels[i], verbose=0)
            decentralized_metrics[model.name] = {'acc': acc, 'auc': auc, "f1": f1, 'loss': loss, 'mse': mse}
            logger.info(
                f"Model{i} decentralized evaluation: accuracy: {acc} - AUC: {auc} - F1: {f1} - loss: {loss} - MSE {mse}")

            logger.info(f"Updating ensemble with model model{i}")
            ensemble = set_base_learner(ensemble=ensemble, bl_name=model.name, bl_weights=model.get_weights())

        losse, acce, msee, auce, f1e = evaluate_model(ensemble, test_data, test_labels[0], verbose=0)
        decentralized_metrics[ensemble.name] = {'acc': acce, 'auc': auce, "f1": f1e, 'loss': losse, 'mse': msee}
        logger.info(
            f"Ensemble decentralized evaluation: accuracy: {acce} - AUC: {auce} - F1: {f1e} - loss: {losse} - MSE {msee}")
        round_metrics["decentralized"] = decentralized_metrics

        # Entrenamiento y evaluación centralizada
        logger.info(f"Training ensemble - round {r}")
        ensemble.fit(train_data, train_labels[0], epochs=2, batch_size=64, verbose=0,
                     use_multiprocessing=True, workers=-1)
        losse, acce, msee, auce, f1e = evaluate_model(ensemble, test_data, test_labels[0], verbose=0)
        logger.info(
            f"Ensemble centralized evaluation: accuracy: {acce} - AUC: {auce} - F1: {f1e} - loss: {losse} - MSE {msee}")
        # TODO Ver por que las metricas centralizadas en los modelos dan los mismos resultados
        for i in range(num_parties):
            model = models[i]
            loss, acc, mse, auc, f1 = evaluate_model(model, test_data[i], test_labels[i], verbose=0)
            centralized_metrics[models[i].name] = {'acc': acc, 'auc': auc, "f1": f1, 'loss': loss, 'mse': mse}
            logger.info(
                f"Model{i} centralized evaluation: accuracy: {acc} - AUC: {auc} - F1: {f1} - loss: {loss} - MSE {mse}")

        centralized_metrics[ensemble.name] = {'acc': acce, 'auc': auce, "f1": f1e, 'loss': losse, 'mse': msee}
        # Se guardan las metricas del ensemble al final del entrenamiento de los modelos para mantener orden.
        round_metrics["centralized"] = centralized_metrics
        metrics[r] = round_metrics
    logger.info("Train finished")

    return metrics


def launch_experiment(num_parties: int = 2, model_type: str = "simple", rounds: int = 2, used_dataset: str = None,
                      experiment_name: str = None, shuffle: bool = False):
    data_preproc, data_labels = preprocess_data(name=used_dataset, shuffle=shuffle, testing=False)
    metrics = execute_experiment(num_parties=num_parties, data_preproc=data_preproc, data_labels=data_labels,
                                 rounds=rounds,
                                 model_type=model_type)
    exp_params = {"experiment_name": experiment_name, "num_parties": num_parties, "model_type": model_type,
                  "rounds": rounds, "used_dataset": used_dataset, "shuffle": shuffle}

    result = {"experiment_params": exp_params, "experiment_metrics": metrics}
    global exp_name
    exp_name = save_experiment_results(experiment_name=experiment_name, experiment_metrics=result)


def read_args():
    parser = argparse.ArgumentParser(description='Parameters for running the experiment')
    parser.add_argument('-np', '--num_parties', type=int, default=2, help='Number of parties to simulate. By default 2')
    parser.add_argument('-m', '--model_type', type=str, default="simple",
                        help='Type of model to use: "simple" or "complex". By default "simple"')
    parser.add_argument('-r', '--rounds', type=int, default=2, help='Number of rounds. By default 2')
    parser.add_argument('-d', '--used_dataset', type=str, default=None,
                        help='Dataset to use: "breast_cancer" or "adult_income". By default random dataset')
    parser.add_argument('-n', '--experiment_name', type=str, default="experiment",
                        help='Name of the experiment. By default "experiment"')
    parser.add_argument('-s', '--shuffle', type=bool, default=False, help='Shuffle the dataset. By default False')
    parser.add_argument('-v', '--view', type=bool, default=True,
                        help='Show the report view of the experiment. By default True')
    args = parser.parse_args()
    print(args.shuffle)
    return args


def process_args():
    args = read_args()
    logger.info("---- New execution ----")
    start = time.time()
    logger.info("Launch configuration:")
    logger.info(f"Number of parties: {args.num_parties}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Number of rounds: {args.rounds}")
    logger.info(f"Used dataset: {args.used_dataset}")
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Shuffle: {args.shuffle}")
    # global exp_name
    # exp_name = args.experiment_name
    dataset = args.used_dataset if not None else random.choice(["breast_cancer", "adult_income"])
    launch_experiment(num_parties=args.num_parties, model_type=args.model_type, rounds=args.rounds,
                      used_dataset=dataset, experiment_name=args.experiment_name, shuffle=args.shuffle)
    logger.info(f"Execution time: {round(time.time() - start, 2)} s.")
    if args.view:
        print_json()



def print_json():
    with open(exp_name, "r") as f:
        data = json.load(f)

    print("Experiment Report:")
    print(json.dumps(data, indent=4))


if __name__ == '__main__':
    process_args()
