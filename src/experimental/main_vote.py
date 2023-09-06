from sklearn.model_selection import train_test_split

from src.datasets import preprocess_data_adult_income, load_client_data
from src.models import load_simple_model_with_more_params
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_auc_score


# def vote_ensemble(models, inputs):
#     predictions = []
#     for i in range(len(models)):
#         predictions.append(models[i].predict(inputs[i]))
#
#     # Combina las predicciones utilizando un proceso de votación.
#     votes = np.mean(predictions, axis=0)
#     new_array = np.where(votes > 0.5, 1, 0)
#     print(new_array)
#     print(new_array.shape)
#     return new_array

def vote_ensemble(predictions):
    """Combina las predicciones de varios modelos.

    Args:
      predictions: Una lista de predicciones.

    Returns:
      La predicción combinada de los modelos.
    """
    # votes = np.argmax(predictions, axis=1)
    # votes = np.bincount(votes)
    # prediction = np.argmax(votes)
    votes = []
    for prediction in predictions:
        prediction = np.array(prediction)
        votes.append(np.where(prediction >= 0.5, 1, 0))
    # votes = np.where(predictions >= 0.5, 1, 0)
    # print(f"Votes shape: {len(votes)}")
    # decision = []
    # for i in range(len(votes[1])):
    #     values = votes[0][i] + votes[1][i] + votes[2][i] + votes[3][i] + votes[4][i]
    #     print(values)
    #     if values >= 3:
    #         decision.append(1)
    #     else:
    #         decision.append(0)
    # print(f"Votes shape: {votes.shape}")
    votes = np.mean(predictions, axis=0)
    votes = np.where(votes >= 0.5, 1, 0)

    return votes


if __name__ == '__main__':
    print('Loading data')
    data_preproc, data_labels = preprocess_data_adult_income(shuffle=False, testing=False)

    # Split data
    print('Splitting data')
    data1, labels1 = load_client_data(num_parties=3, client_id=1, data=data_preproc, labels=data_labels)
    data1 = data1[:, 1:]
    train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)

    data2, labels2 = load_client_data(num_parties=3, client_id=2, data=data_preproc, labels=data_labels)
    data2 = data2[:, 1:]
    train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)

    data3, labels3 = load_client_data(num_parties=3, client_id=3, data=data_preproc, labels=data_labels)
    data3 = data3[:, 1:]
    train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)

    # data4, labels4 = load_client_data(num_parties=5, client_id=4, data=data_preproc, labels=data_labels)
    # data4 = data4[:, 1:]
    # train4, test4, train_labels4, test_labels4 = train_test_split(data4, labels4, test_size=0.2, random_state=42)
    #
    # data5, labels5 = load_client_data(num_parties=5, client_id=5, data=data_preproc, labels=data_labels)
    # data5 = data5[:, 1:]
    # train5, test5, train_labels5, test_labels5 = train_test_split(data5, labels5, test_size=0.2, random_state=42)

    print('Loading models')
    model1 = load_simple_model_with_more_params(data1.shape[1], 'model1')
    model2 = load_simple_model_with_more_params(data2.shape[1], 'model2')
    model3 = load_simple_model_with_more_params(data3.shape[1], 'model3')
    # model4 = load_simple_model2(data4.shape[1], 'model4')
    # model5 = load_simple_model2(data5.shape[1], 'model5')

    print('Training models')
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    # model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    # model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    model1.fit(data1, labels1, epochs=5, batch_size=128, verbose=0)
    model2.fit(data2, labels2, epochs=5, batch_size=128, verbose=0)
    model3.fit(data3, labels3, epochs=5, batch_size=128, verbose=0)
    # model4.fit(data4, labels4, epochs=5, batch_size=128, verbose=0)
    # model5.fit(data5, labels5, epochs=5, batch_size=128, verbose=0)

    print('Evaluation of models')
    model1.evaluate(test1, test_labels1)
    model2.evaluate(test2, test_labels2)
    model3.evaluate(test3, test_labels3)
    # model4.evaluate(test4, test_labels4)
    # model5.evaluate(test5, test_labels5)

    predictions1 = model1.predict(test1)
    predictions2 = model2.predict(test2)
    predictions3 = model3.predict(test3)
    # predictions4 = model4.predict(test4)
    # predictions5 = model5.predict(test5)
    print(f"Roc for model1 {roc_auc_score(test_labels1, predictions1[:, 0])}")
    print(f"Roc for model2 {roc_auc_score(test_labels2, predictions2[:, 0])}")
    print(f"Roc for model3 {roc_auc_score(test_labels3, predictions3[:, 0])}")
    # print(f"Roc for model4 {roc_auc_score(test_labels4, predictions4[:, 0])}")
    # print(f"Roc for model5 {roc_auc_score(test_labels5, predictions5[:, 0])}")

    predictions = vote_ensemble(
        [predictions1[:, 0], predictions2[:, 0], predictions3[:, 0]])
    # unique, counts = np.unique(predictions, return_counts=True)
    # print(dict(zip(unique, counts)))
    print(accuracy_score(test_labels1, np.array(predictions)))
    print(roc_auc_score(test_labels1, np.array(predictions)))

    # ensemble = vote_ensemble([model1, model2, model3], [test1, test2, test3])
