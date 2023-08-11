import pytest
from src.models import load_simple_model, load_simple_model_with_more_params, load_ensemble, set_base_learner, \
    get_base_learner
from tensorflow import keras
import random


def test_load_simple_model():
    input_dim = 10
    model = load_simple_model(input_dim=input_dim, name='simple_model')
    # Caso 1: Instancia de keras.Model
    assert isinstance(model, keras.Model)

    # Caso 2: Arquitectura correcta
    assert len(model.layers) == 5  # 4 capas + input
    assert model.layers[1].units == input_dim
    assert model.layers[2].units == 8
    assert model.layers[3].units == 4
    assert model.layers[4].units == 1

    # Caso 3: Nombre correcto
    assert model.name == "simple_model"

    # Caso 4: Activaciones correctas
    # assert model.layers[0].activation == "selu"
    assert model.layers[1].activation == keras.activations.selu
    assert model.layers[2].activation == keras.activations.selu
    assert model.layers[3].activation == keras.activations.selu
    assert model.layers[4].activation == keras.activations.sigmoid

    # Caso 5:  Comprueba I/O
    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)
