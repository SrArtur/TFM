import math

import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# TODO Comentar
# TODO Logs
def preprocess_data(shuffle: bool = False, testing: bool = False):
    try:
        if testing:
            data = pd.read_csv("data/adult.data")
        else:
            print("Loading dataset from local file...")
            data = pd.read_csv(config.adult_data_path)
    except FileNotFoundError:
        print("Dataset not found. Downloading from UCI repository...")
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

