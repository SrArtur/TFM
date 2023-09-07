# TFM ![example workflow](https://github.com/srartur/tfm/actions/workflows/blank.yml/badge.svg)
**Repositorio del Trabajo Final de Máster de Arturo Ortiz**

Se presenta una propuesta de Agregador Basado en Stacking para Aprendizaje Federado Vertical. Para más detalles consultar la memoria del trabajo.

## Argumentos para ejecutar experimentos
````bash

usage: main.py [-h] [-np NUM_PARTIES] [-m MODEL_TYPE] [-r ROUNDS] [-d USED_DATASET]
               [-n EXPERIMENT_NAME] [-s SHUFFLE] [-v VIEW]

Parameters for running the experiment

options:
  -h, --help            show this help message and exit
  -np NUM_PARTIES, --num_parties NUM_PARTIES
                        Number of parties to simulate. By default 2
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        Type of model to use: "simple" or "complex". By default "simple"
  -r ROUNDS, --rounds ROUNDS
                        Number of rounds. By default 2
  -d USED_DATASET, --used_dataset USED_DATASET
                        Dataset to use: "breast_cancer" or "adult_income". By default random dataset   
  -n EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        Name of the experiment. By default "experiment"
  -s SHUFFLE, --shuffle SHUFFLE
                        Shuffle the dataset. By default False
  -v VIEW, --view VIEW  Show the report view of the experiment. By default True


````

## Instalación de dependencias
````bash
pip install -r requirements.txt
````

## Ejemplo de ejecución (local):
Dos participantes, modelo simple, dos rondas, conjunto de datos: breast_cancer, nombre del experimento hello_vfl
````bash
python main.py -np 2 -m complex -r 2 -d breast_cancer -n hello_vfl
````

## Construcción de la imagen Docker
````bash
docker build -t tfm .
````

## Ejemplo de ejecución (Docker):
Dos participantes, modelo simple, dos rondas, conjunto de datos: breast_cancer, nombre del experimento hello_vfl
````bash
docker run tfm -d breast_cancer -m complex -np 2 -r 2 -n hello_vfl
````