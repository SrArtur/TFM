import keras.backend
import optuna
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
import optuna_integration

from src.datasets import preprocess_data_adult_income, load_client_data, preprocess_data
from src.models import load_simple_model, load_simple_model_with_more_params


def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 5, log=True)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 1, 128, log=True)
        model.add(Dense(num_hidden, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=["accuracy", "mse"],
    )

    return model


def objective(trial):
    keras.backend.clear_session()

    # model = create_model(trial)
    model = load_simple_model_with_more_params(train3.shape[1], "simple_model")
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=["accuracy", "mse"],
    )
    epochs = trial.suggest_int("epochs", 1, 10)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    model.fit(train3, train_labels3, epochs=epochs, batch_size=batch_size, verbose=0,
              validation_split=0.2)

    score = model.evaluate(test3, test_labels3, verbose=1)
    return score[1]


if __name__ == '__main__':
    data_preproc, data_labels = preprocess_data(name="breast_cancer", shuffle=False, testing=False)

    # data1, labels1 = load_client_data(num_parties=3, client_id=1, data=data_preproc, labels=data_labels)
    # data1 = data1[:, 1:]
    # train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)

    # data2, labels2 = load_client_data(num_parties=3, client_id=2, data=data_preproc, labels=data_labels)
    # data2 = data2[:, 1:]
    # train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)

    data3, labels3 = load_client_data(num_parties=3, client_id=3, data=data_preproc, labels=data_labels)
    data3 = data3[:, 1:]
    train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)


    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
