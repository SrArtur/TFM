import math

import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger("TFM")


def preprocess_data(name: str = None, shuffle: bool = False, testing: bool = False, ):
    if name == "adult_income":
        return preprocess_data_adult_income(shuffle, testing)
    elif name == "breast_cancer":
        return preprocess_data_breast_cancer(shuffle, testing)
    else:
        raise ValueError("Dataset name not found")


# TODO Comentar
def preprocess_data_breast_cancer(shuffle: bool = False, testing: bool = False):
    try:
        if testing:
            data = pd.read_csv("data/breast-cancer.data")
        else:
            logger.info(f"Loading dataset BREAST CANCER from local file...")
            data = pd.read_csv(config.breast_cancer_data_path)
    except FileNotFoundError:
        logger.info("Dataset BREAST CANCER not found. Downloading from UCI repository...")
        data = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
        data.to_csv(config.breast_cancer_data_path, index=False)
    data.columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
                    'fractal_dimension_worst']
    data = data.replace('?', np.nan)
    data = data.dropna()

    if shuffle:
        target = data['diagnosis']
        data = data.iloc[:, 1:]
        data = data.sample(frac=1, axis=1).reset_index(drop=True)
        data['diagnosis'] = target

    data = data.drop(['id'], axis=1)
    data_labels = data['diagnosis']
    data_labels = data_labels.replace({'M': 1, 'B': 0})

    sc = StandardScaler()
    data_scaled = sc.fit_transform(data.drop('diagnosis', axis=1))
    id_column = np.arange(0, len(data_scaled))
    data_scaled = np.insert(data_scaled, 0, id_column, axis=1)
    data_labels = np.column_stack((id_column, data_labels))

    return data_scaled, data_labels


def preprocess_data_adult_income(shuffle: bool = False, testing: bool = False):
    try:
        if testing:
            data = pd.read_csv("data/adult.data")
        else:
            logger.info("Loading ADULT INCOME dataset from local file...")
            data = pd.read_csv(config.adult_data_path)
    except FileNotFoundError:
        logger.info("Dataset BREAST CANCER not found. Downloading from UCI repository...")
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
        data.to_csv(config.adult_data_path, index=False)
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = data.replace(' ?', np.nan)
    data = data.dropna()

    if shuffle:
        target = data['income']
        data = data.iloc[:, :-1]
        data = data.sample(frac=1, axis=1).reset_index(drop=True)
        data['income'] = target

    data_encoded = data.apply(LabelEncoder().fit_transform)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded.drop('income', axis=1))
    id_colum = np.arange(0, len(data_scaled))
    data_scaled = np.insert(data_scaled, 0, id_colum, axis=1)
    data_labels = np.column_stack((id_colum, data_encoded['income']))

    return data_scaled, data_labels


def load_client_data(num_parties, client_id, data, labels):
    # TODO Poner si devolver con o sin train_test_split
    # TODO Poner si se devuelve con o sin ID
    num_chars = len(data[0]) - 1  # Para evitar considerar el ID como característica
    chars_per_party = math.floor(num_chars / num_parties)
    data = data[:, 1:]
    if num_parties >= num_chars or client_id - 1 >= num_parties:
        raise ValueError(
            "Number of parties must be less or equal than the number of characteristics or Client id must be less "
            "than number of parties")
    elif client_id == 0:
        raise ValueError("Client id must be greater than 0")
    elif client_id == num_parties:
        party_charts = data[:, (client_id - 1) * chars_per_party:]
    else:
        party_charts = data[:, (client_id - 1) * chars_per_party: client_id * chars_per_party]

    return np.hstack((labels[:, 0].reshape(-1, 1), party_charts)), labels[:, 1].reshape(-1, 1)
