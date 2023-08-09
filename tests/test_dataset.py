import numpy as np
from src.datasets import preprocess_data_adult_income, load_client_data
import pytest
import os
import sys


def test_preprocess_data_adult_income():
    # Ejecutar la función sin shuffle
    data_scaled, data_labels = preprocess_data_adult_income(shuffle=False, testing=True)

    # Comprobar que los datos tienen el formato esperado
    assert isinstance(data_scaled, np.ndarray)
    assert isinstance(data_labels, np.ndarray)

    # Comprobar que no hay valores faltantes en los datos
    assert np.isnan(data_scaled).sum() == 0

    # Comprobar que las etiquetas están codificadas correctamente
    assert np.unique(data_labels[1]).size == 2

    # Ejecutar la función con shuffle
    data_scaled_shuffled, data_labels_shuffled = preprocess_data_adult_income(shuffle=True, testing=True)

    # Comprobar que los datos tienen el formato esperado
    assert isinstance(data_scaled_shuffled, np.ndarray)
    assert isinstance(data_labels_shuffled, np.ndarray)

    # Comprobar que no hay valores faltantes en los datos
    assert np.isnan(data_scaled_shuffled).sum() == 0

    # Comprobar que las etiquetas están codificadas correctamente
    assert np.unique(data_labels_shuffled[1]).size == 2

    # Comprobar que los datos están realmente mezclados
    assert not np.array_equal(data_scaled, data_scaled_shuffled)
    assert not np.array_equal(data_labels, data_labels_shuffled)


def test_preprocess_data_breast_cancer():
    # Ejecutar la función sin shuffle
    data_scaled, data_labels = preprocess_data_adult_income(shuffle=False, testing=True)

    # Comprobar que los datos tienen el formato esperado
    assert isinstance(data_scaled, np.ndarray)
    assert isinstance(data_labels, np.ndarray)

    # Comprobar que no hay valores faltantes en los datos
    assert np.isnan(data_scaled).sum() == 0

    # Comprobar que las etiquetas están codificadas correctamente
    assert np.unique(data_labels[1]).size == 2

    # Ejecutar la función con shuffle
    data_scaled_shuffled, data_labels_shuffled = preprocess_data_adult_income(shuffle=True, testing=True)

    # Comprobar que los datos tienen el formato esperado
    assert isinstance(data_scaled_shuffled, np.ndarray)
    assert isinstance(data_labels_shuffled, np.ndarray)

    # Comprobar que no hay valores faltantes en los datos
    assert np.isnan(data_scaled_shuffled).sum() == 0

    # Comprobar que las etiquetas están codificadas correctamente
    assert np.unique(data_labels_shuffled[1]).size == 2

    # Comprobar que los datos están realmente mezclados
    assert not np.array_equal(data_scaled, data_scaled_shuffled)
    assert not np.array_equal(data_labels, data_labels_shuffled)


@pytest.fixture
def sample_data_preprocessed():
    # Datos de ejemplo para las pruebas
    data = np.array([[0, 1, 2, 3, 4, 5],
                     [1, 6, 7, 8, 9, 10],
                     [2, 11, 12, 13, 14, 15],
                     [3, 16, 17, 18, 19, 20]])
    labels = np.array([[0, 1],
                       [1, 0],
                       [2, 1],
                       [3, 0]])
    return data, labels


def test_load_client_data(sample_data_preprocessed):
    data, labels = sample_data_preprocessed

    # Caso de prueba 1: Número de partes igual a 2 y ID de cliente igual a 1
    num_parties = 2
    client_id = 1
    expected_data = np.array([[0, 1, 2],
                              [1, 6, 7],
                              [2, 11, 12],
                              [3, 16, 17]])
    expected_labels = np.array([[1], [0], [1], [0]])
    data_loaded, labels_loaded = load_client_data(num_parties, client_id, data, labels)
    assert np.array_equal(data_loaded, expected_data)
    assert np.array_equal(labels_loaded, expected_labels)

    # Caso de prueba 2: Número de partes igual a 3 y ID de cliente igual a 3
    num_parties = 3
    client_id = 3
    expected_data = np.array([[0, 3, 4, 5],
                              [1, 8, 9, 10],
                              [2, 13, 14, 15],
                              [3, 18, 19, 20]])
    expected_labels = np.array([[1],
                                [0],
                                [1],
                                [0]])
    data_loaded, labels_loaded = load_client_data(num_parties, client_id, data, labels)
    assert np.array_equal(data_loaded, expected_data)
    assert np.array_equal(labels_loaded, expected_labels)
    # Caso de prueba 3: Número de partes igual a 4 y ID de cliente igual a 2
    num_parties = 4
    client_id = 2
    expected_data = np.array([[0, 2],
                              [1, 7],
                              [2, 12],
                              [3, 17]])
    expected_labels = np.array([[1],
                                [0],
                                [1],
                                [0]])
    data_loaded, labels_loaded = load_client_data(num_parties, client_id, data, labels)
    assert np.array_equal(data_loaded, expected_data)
    assert np.array_equal(labels_loaded, expected_labels)

    # Caso de prueba 4: Número de partes igual a 4 y ID de cliente igual a 0
    num_parties = 4
    client_id = 0
    with pytest.raises(ValueError):
        load_client_data(num_parties, client_id, data, labels)

    # Caso de prueba 5: Número de partes igual a 3 y ID de cliente igual a 4
    num_parties = 3
    client_id = 4
    with pytest.raises(ValueError):
        load_client_data(num_parties, client_id, data, labels)

    # Caso de prueba 6: Número de partes igual a 2 y ID de cliente igual a 2
    num_parties = 2
    client_id = 2
    with pytest.raises(ValueError):
        load_client_data(num_parties, client_id, data[:1], labels)
