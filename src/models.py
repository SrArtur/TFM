import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from keras.layers import Dense, Input, Concatenate
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)


def load_simple_model(input_dim: int, name: str):
    inputs = Input(shape=(input_dim,))
    layer = Dense(input_dim, activation='selu')(inputs)
    layer = Dense(8, activation='selu')(layer)
    layer = Dense(4, activation='selu')(layer)
    layer = Dense(1, activation='sigmoid')(layer)  # Cambiar a 'linear' si logits=True
    return keras.Model(inputs, layer, name=name)


def load_simple_model2(input_dim: int, name: str):
    inputs = Input(shape=(input_dim,))
    layer = Dense(input_dim, activation='selu')(inputs)
    layer = Dense(128, activation='selu')(layer)
    layer = Dense(64, activation='selu')(layer)
    layer = Dense(1, activation='sigmoid')(layer)  # Cambiar a 'linear' si logits=True
    return keras.Model(inputs, layer, name=name)


def load_ensemble(models: list):
    inputs = [m.input.shape[1] for m in models]
    inputs = {'input_' + str(i): Input(shape=(inputs[i]), name='input_' + str(i)) for i in range(len(inputs))}
    output = {'output_' + str(i): models[i](inputs['input_' + str(i)]) for i in range(len(models))}
    ensemble = Concatenate()(list(output.values()))
    ensemble = Dense(6, activation='selu')(ensemble)
    ensemble = Dense(1, activation='sigmoid')(ensemble)
    return keras.Model(inputs=inputs, outputs=ensemble, name='ensemble')


def get_base_learner(ensemble, model_name: str):
    model_layer = [layer for layer in ensemble.layers if layer.name == model_name][0]
    return model_layer.weights


def set_base_learner(ensemble, bl_name: str, bl_weights: list):
    print("Setting weights for ensemble")
    for i in range(len(ensemble.layers)):
        if ensemble.layers[i].name == bl_name:
            if not np.array_equal(ensemble.layers[i].weights, bl_weights):
                print(f"Base learner {bl_name} weights have been updated")
            else:
                print(f"Base learner {bl_name} weights have not been updated")
            ensemble.layers[i].set_weights(bl_weights)
            break
    return ensemble
