from src.models import load_simple_model, load_simple_model_with_more_params, load_ensemble, set_base_learner, \
    get_base_learner
import logging

import numpy as np
from tensorflow import keras

from src.models import load_simple_model, load_simple_model_with_more_params, load_ensemble, set_base_learner, \
    get_base_learner

logger = logging.getLogger("TFM")


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


def test_load_simple_modelo_with_more_params():
    input_dim = 10
    model = load_simple_model_with_more_params(input_dim=input_dim, name='simple_model')
    # Caso 1: Instancia de keras.Model
    assert isinstance(model, keras.Model)

    # Caso 2: Arquitectura correcta
    assert len(model.layers) == 5  # 4 capas + input
    assert model.layers[1].units == input_dim
    assert model.layers[2].units == 128
    assert model.layers[3].units == 64
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


def test_load_ensemble():
    # Caso 1: Ensemble con modelo simple
    input_dim = 10
    name = "simple_model_"
    num_models = 6
    models = [load_simple_model(input_dim=input_dim, name=name + str(i)) for i in range(num_models)]
    ensemble = load_ensemble(models=models)

    # Caso 1.1: Instancia de keras.Model y nombre correcto
    assert isinstance(ensemble, keras.Model)
    assert ensemble.name == "ensemble"

    # Caso 1.2: Arquitectura correcta
    assert len(ensemble.layers) == 3 + num_models + len(models)  # 1 capa + input + modelos
    for i in range(len(models)):  # Cada uno de los Inputs del ensemble deben tener el mismo input que los modelos base
        assert ensemble.layers[i].input_shape[0][1] == input_dim

    # Caso 1.3: Capa de activación correcta
    assert ensemble.layers[-1].activation == keras.activations.sigmoid

    # Caso 1.4: Comprueba I/O
    assert ensemble.input_shape == {"input_" + str(i): (None, input_dim) for i in range(num_models)}
    assert ensemble.output_shape == (None, 1)


# TODO Mirar como hacer el resto de comprobaciones de actualizaciones del modelo
# TODO Pensar en tratamiento de excepciones para las creaciciones y funciones del modelo
def test_update_weights():
    model1 = load_simple_model(input_dim=10, name='simple_model1')
    model2 = load_simple_model(input_dim=10, name='simple_model2')
    model3 = load_simple_model(input_dim=10, name='simple_model3')
    models = [model1, model2, model3]
    ensemble = load_ensemble(models=[model1, model2, model3])
    for model in models:
        ensemble = set_base_learner(ensemble, model, model.get_weights())
    # for model in models:
    #     assert np.array_equal(get_base_learner(ensemble, model.name), model.get_weights())
