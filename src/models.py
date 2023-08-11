import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.layers import Dense, Input, Concatenate
import os
import numpy as np
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
logger = logging.getLogger("TFM")


def load_simple_model(input_dim: int, name: str):
    """
    Carga un modelo simple con cuatro capas densas y una capa de salida con activación sigmoide.
    :param input_dim: Número de características de entrada. Equivalente a la dimensión de la capa de entrada.
    :param name: Nombre del modelo.
    :return: Modelo keras.
    """
    inputs = Input(shape=(input_dim,))
    layer = Dense(input_dim, activation='selu')(inputs)
    layer = Dense(8, activation='selu')(layer)
    layer = Dense(4, activation='selu')(layer)
    layer = Dense(1, activation='sigmoid')(layer)  # Cambiar a 'linear' si logits=True
    logger.info(f"Loading simple model - {name}")
    return keras.Model(inputs, layer, name=name)


def load_simple_model_with_more_params(input_dim: int, name: str):
    """
    Carga un
    :param input_dim:
    :param name:
    :return:
    """
    inputs = Input(shape=(input_dim,))
    layer = Dense(input_dim, activation='selu')(inputs)
    layer = Dense(128, activation='selu')(layer)
    layer = Dense(64, activation='selu')(layer)
    layer = Dense(1, activation='sigmoid')(layer)  # Cambiar a 'linear' si logits=True
    logger.info(f"Loading complex model - {name}")
    return keras.Model(inputs, layer, name=name)


def load_base_of_ensemble(num_participants: int, name: str):
    _input = Input(shape=(num_participants,))
    ensemble = Dense(6, activation='selu')(_input)
    ensemble = Dense(1, activation='sigmoid')(ensemble)
    logger.info("Loading ensemble with " + str(num_participants) + " models")
    return keras.Model(inputs=_input, outputs=ensemble, name='ensemble')


def load_ensemble(models: list):
    """
    Crea un modelo ensemble basado en stacking con los modelos pasados por parámetro.
    :param models: Lista de modelos keras que compone el ensemble.
    :return: Modelo keras ensemble.
    """
    inputs = [m.input.shape[1] for m in models]
    inputs = {'input_' + str(i): Input(shape=(inputs[i]), name='input_' + str(i)) for i in range(len(inputs))}
    output = {'output_' + str(i): models[i](inputs['input_' + str(i)]) for i in range(len(models))}
    ensemble = Concatenate()(list(output.values()))
    ensemble = Dense(6, activation='selu')(ensemble)
    ensemble = Dense(1, activation='sigmoid')(ensemble)
    logger.info("Loading ensemble with " + str(len(models)) + " models")
    return keras.Model(inputs=inputs, outputs=ensemble, name='ensemble')


def get_base_learner(ensemble, model_name: str):
    """
    Obtiene los pesos del modelo especificado que compone el ensemble.
    :param ensemble: Modelo Keras ensemble.
    :param model_name: Modelo base del ensemble, correspondiente a un cliente.
    :return: Pesos del modelo base.
    """
    model_layer = [layer for layer in ensemble.layers if layer.name == model_name][0]
    logger.info(f"Getting base learner {model_name} weights from ensemble")
    return model_layer.weights


def set_base_learner(ensemble, bl_name: str, bl_weights: list):
    """
    Actualiza los pesos del modelo base especificado que compone el ensemble.
    :param ensemble: Modelo Keras ensemble.
    :param bl_name: Nombre del modelo base del ensemble, correspondiente a un cliente.
    :param bl_weights: Pesos del modelo base, correspondiente a un cliente.
    :return: Modelo Keras ensemble actualizado con los pesos de un modelo base.
    """
    logger.info(f"Updating base learner with {bl_name} weights")
    for i in range(len(ensemble.layers)):
        if ensemble.layers[i].name == bl_name:
            if not np.array_equal(ensemble.layers[i].weights, bl_weights):
                logger.info(f"Base learner {bl_name} weights have been updated")
            else:
                logger.info(f"Base learner {bl_name} weights have not been updated")
            ensemble.layers[i].set_weights(bl_weights)
            break
    return ensemble
