from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from src.datasets import preprocess_data_adult_income, load_client_data, preprocess_data_breast_cancer, preprocess_data, \
    load_client_train_test_split
from src.models import load_simple_model_with_more_params, load_ensemble, get_base_learner, set_base_learner, \
    load_base_of_ensemble, load_simple_model, evaluate_model
from src.experiment_utils import save_experiment_results
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
import config
from tensorflow.keras.utils import plot_model
import time
import os

import logging

logger = logging.getLogger("TFM")
logger.setLevel(logging.INFO)

# def experiment_3_clients():
#     data1, labels1 = load_client_data(num_parties=3, client_id=1, data=data_preproc, labels=data_labels)
#     data1 = data1[:, 1:]
#     train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 1 with shape {data1.shape}")
#
#     data2, labels2 = load_client_data(num_parties=3, client_id=2, data=data_preproc, labels=data_labels)
#     data2 = data2[:, 1:]
#     train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 2 with shape {data2.shape}")
#
#     data3, labels3 = load_client_data(num_parties=3, client_id=3, data=data_preproc, labels=data_labels)
#     data3 = data3[:, 1:]
#     train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 3 with shape {data3.shape}")
#
#     # Load simple model
#
#     model1 = load_simple_model_with_more_params(data1.shape[1], 'model1')
#     model2 = load_simple_model_with_more_params(data2.shape[1], 'model2')
#     model3 = load_simple_model_with_more_params(data3.shape[1], 'model3')
#
#     ensemble = load_ensemble([model1, model2, model3])
#
#     # Compile models
#     model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#     ensemble.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     iterations_num = 1
#     for i in range(iterations_num):
#         logger.info(f'Iteration {str(i + 1)} of {str(iterations_num)}')
#         # Train models
#         logger.info(f'Training model {model1.name}')
#         model1.fit(data1, labels1, epochs=4, batch_size=128, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model2.name}')
#         model2.fit(data2, labels2, epochs=4, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model3.name}')
#         model3.fit(data3, labels3, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#
#         logger.info('Evaluation of models after training')
#         loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         logger.info(
#             f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         logger.info(
#             f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         logger.info(
#             f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")
#
#         set_base_learner(ensemble, "model1", model1.get_weights())
#         set_base_learner(ensemble, "model2", model2.get_weights())
#         set_base_learner(ensemble, "model3", model3.get_weights())
#
#         logger.info('Evaluation of ensemble before training')
#         losse, acce, msee = ensemble.evaluate([test1, test2, test3], test_labels1, verbose=0)
#         predict_ensemble = ensemble.predict([test1, test2, test3], verbose=0)
#         roce = roc_auc_score(test_labels1, predict_ensemble)
#         logger.info(
#             f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")
#         logger.info("Training ensemble")
#         ensemble.fit([train1, train2, train3], train_labels1, epochs=4, batch_size=128, verbose=0,
#                      use_multiprocessing=True, workers=-1)
#
#         logger.info('Evaluation of ensemble after training')
#         losse, acce, msee = ensemble.evaluate([test1, test2, test3], test_labels1, verbose=0)
#         predict_ensemble = ensemble.predict([test1, test2, test3], verbose=0)
#         roce = roc_auc_score(test_labels1, predict_ensemble)
#         logger.info(
#             f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")
#
#         logger.info('Setting weights for base learners')
#         model1 = load_simple_model_with_more_params(data1.shape[1], 'model1')
#         model2 = load_simple_model_with_more_params(data2.shape[1], 'model2')
#         model3 = load_simple_model_with_more_params(data3.shape[1], 'model3')
#
#         model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#         bweights_model1 = model1.get_weights()
#         model1.set_weights(get_base_learner(ensemble, "model1"))
#         if not np.array_equal(bweights_model1, model1.get_weights()):
#             logger.info(f'Weights are updated for model1')
#         bweights_model2 = model2.get_weights()
#         model2.set_weights(get_base_learner(ensemble, "model2"))
#         if not np.array_equal(bweights_model2, model2.get_weights()):
#             logger.info(f'Weights are updated for model2')
#         bweights_model3 = model3.get_weights()
#         model3.set_weights(get_base_learner(ensemble, "model3"))
#         if not np.array_equal(bweights_model3, model3.get_weights()):
#             logger.info(f'Weights are updated for model3')
#
#         logger.info('Evaluation of models after setting weights and no training')
#         loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         logger.info(
#             f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         logger.info(
#             f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         logger.info(
#             f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")
#         # logger.info(f'Training model {model1.name}')
#         # model1.fit(data1, labels1, epochs=4, batch_size=128, verbose=0, use_multiprocessing=True, workers=-1)
#         # logger.info(f'Training model {model2.name}')
#         # model2.fit(data2, labels2, epochs=4, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         # logger.info(f'Training model {model3.name}')
#         # model3.fit(data3, labels3, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#
#         # logger.info('Evaluation of models after training')
#         # loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         # roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         # logger.info(
#         #     f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         # loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         # roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         # logger.info(
#         #     f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         # loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         # roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         # logger.info(
#         #     f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")


# def experiments_6_clients():
#     # TODO Automatizar la creacion de clientes
#     data1, labels1 = load_client_data(num_parties=6, client_id=1, data=data_preproc, labels=data_labels)
#     data1 = data1[:, 1:]
#     train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 1 with shape {data1.shape}")
#
#     data2, labels2 = load_client_data(num_parties=6, client_id=2, data=data_preproc, labels=data_labels)
#     data2 = data2[:, 1:]
#     train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 2 with shape {data2.shape}")
#
#     data3, labels3 = load_client_data(num_parties=6, client_id=3, data=data_preproc, labels=data_labels)
#     data3 = data3[:, 1:]
#     train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 3 with shape {data3.shape}")
#
#     data4, labels4 = load_client_data(num_parties=6, client_id=4, data=data_preproc, labels=data_labels)
#     data4 = data4[:, 1:]
#     train4, test4, train_labels4, test_labels4 = train_test_split(data4, labels4, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 4 with shape {data4.shape}")
#
#     data5, labels5 = load_client_data(num_parties=6, client_id=5, data=data_preproc, labels=data_labels)
#     data5 = data5[:, 1:]
#     train5, test5, train_labels5, test_labels5 = train_test_split(data5, labels5, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 5 with shape {data5.shape}")
#
#     # data6, labels6 = load_client_data(num_parties=6, client_id=6, data=data_preproc, labels=data_labels)
#     # data6 = data6[:, 1:]
#     # train6, test6, train_labels6, test_labels6 = train_test_split(data6, labels6, test_size=0.2, random_state=42)
#     # logger.info(f"Splitting data for client 6 with shape {data6.shape}")
#
#     # Load simple model
#
#     model1 = load_simple_model(data1.shape[1], 'model1')
#     model2 = load_simple_model(data2.shape[1], 'model2')
#     model3 = load_simple_model(data3.shape[1], 'model3')
#     model4 = load_simple_model(data4.shape[1], 'model4')
#     model5 = load_simple_model(data5.shape[1], 'model5')
#     # model6 = load_simple_model(data6.shape[1], 'model6')
#
#     ensemble = load_ensemble([model1, model2, model3, model4, model5])
#
#     # Compile models
#     model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     # model6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#     ensemble.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#     iterations_num = 1
#     for i in range(iterations_num):
#         logger.info(f'Iteration {str(i)} of {str(iterations_num)}')
#         # Train models
#         logger.info(f'Training model {model1.name}')
#         model1.fit(data1, labels1, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model2.name}')
#         model2.fit(data2, labels2, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model3.name}')
#         model3.fit(data3, labels3, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model4.name}')
#         model4.fit(data4, labels4, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model5.name}')
#         model5.fit(data5, labels5, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         # logger.info(f'Training model {model6.name}')
#         # model6.fit(data6, labels6, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#
#         logger.info('Evaluation of models after training')
#         loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         logger.info(
#             f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         logger.info(
#             f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         logger.info(
#             f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")
#         loss4, acc4, mse4 = model4.evaluate(test4, test_labels4, verbose=0)
#         roc4 = roc_auc_score(test_labels1, model4.predict(test4, verbose=0))
#         logger.info(
#             f"Model 4 evaluation: roc {round(roc4, 4)}, accuracy {round(acc4, 4)}, loss: {round(loss4, 4)}, mse {round(mse4, 4)}")
#         loss5, acc5, mse5 = model5.evaluate(test5, test_labels5, verbose=0)
#         roc5 = roc_auc_score(test_labels1, model5.predict(test5, verbose=0))
#         logger.info(
#             f"Model 5 evaluation: roc {round(roc5, 4)}, accuracy {round(acc5, 4)}, loss: {round(loss5, 4)}, mse {round(mse5, 4)}")
#         # loss6, acc6, mse6 = model6.evaluate(test6, test_labels6, verbose=0)
#         # roc6 = roc_auc_score(test_labels1, model6.predict(test6, verbose=0))
#         # logger.info(
#         #     f"Model 6 evaluation: roc {round(roc6, 4)}, accuracy {round(acc6, 4)}, loss: {round(loss6, 4)}, mse {round(mse6, 4)}")
#
#         set_base_learner(ensemble, "model1", model1.get_weights())
#         set_base_learner(ensemble, "model2", model2.get_weights())
#         set_base_learner(ensemble, "model3", model3.get_weights())
#         set_base_learner(ensemble, "model4", model4.get_weights())
#         set_base_learner(ensemble, "model5", model5.get_weights())
#
#         logger.info('Evaluation of ensemble before training')
#         losse, acce, msee = ensemble.evaluate([test1, test2, test3, test4, test5], test_labels1, verbose=0)
#         predict_ensemble = ensemble.predict([test1, test2, test3, test4, test5], verbose=0)
#         roce = roc_auc_score(test_labels1, predict_ensemble)
#         logger.info(
#             f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")
#         logger.info("Training ensemble")
#         ensemble.fit([train1, train2, train3, train4, train5], train_labels1, epochs=4, batch_size=128,
#                      verbose=0,
#                      use_multiprocessing=True, workers=-1)
#
#         logger.info('Evaluation of ensemble after training')
#         losse, acce, msee = ensemble.evaluate([test1, test2, test3, test4, test5], test_labels1, verbose=0)
#         predict_ensemble = ensemble.predict([test1, test2, test3, test4, test5], verbose=0)
#         roce = roc_auc_score(test_labels1, predict_ensemble)
#         logger.info(
#             f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")
#
#         logger.info('Setting weights for base learners')
#         model1 = load_simple_model(data1.shape[1], 'model1')
#         model2 = load_simple_model(data2.shape[1], 'model2')
#         model3 = load_simple_model(data3.shape[1], 'model3')
#         model4 = load_simple_model(data4.shape[1], 'model4')
#         model5 = load_simple_model(data5.shape[1], 'model5')
#
#         model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#         bweights_model1 = model1.get_weights()
#         model1.set_weights(get_base_learner(ensemble, "model1"))
#         if not np.array_equal(bweights_model1, model1.get_weights()):
#             logger.info(f'Weights are updated for model1')
#         bweights_model2 = model2.get_weights()
#         model2.set_weights(get_base_learner(ensemble, "model2"))
#         if not np.array_equal(bweights_model2, model2.get_weights()):
#             logger.info(f'Weights are updated for model2')
#         bweights_model3 = model3.get_weights()
#         model3.set_weights(get_base_learner(ensemble, "model3"))
#         if not np.array_equal(bweights_model3, model3.get_weights()):
#             logger.info(f'Weights are updated for model3')
#         bweights_model4 = model4.get_weights()
#         model4.set_weights(get_base_learner(ensemble, "model4"))
#         if not np.array_equal(bweights_model4, model4.get_weights()):
#             logger.info(f'Weights are updated for model4')
#         bweights_model5 = model5.get_weights()
#         model5.set_weights(get_base_learner(ensemble, "model5"))
#         if not np.array_equal(bweights_model5, model5.get_weights()):
#             logger.info(f'Weights are updated for model5')
#
#         logger.info('Evaluation of models after setting weights and no training')
#         loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         logger.info(
#             f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         logger.info(
#             f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         logger.info(
#             f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")
#         logger.info(f'Training model {model1.name}')
#         model1.fit(data1, labels1, epochs=4, batch_size=128, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model2.name}')
#         model2.fit(data2, labels2, epochs=4, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model3.name}')
#         model3.fit(data3, labels3, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model4.name}')
#         model4.fit(data4, labels4, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model5.name}')
#         model5.fit(data5, labels5, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#
#         logger.info('Evaluation of models after training')
#         loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
#         roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
#         logger.info(
#             f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
#         loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
#         roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
#         logger.info(
#             f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
#         loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
#         roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
#         logger.info(
#             f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")
#         loss4, acc4, mse4 = model4.evaluate(test4, test_labels4, verbose=0)
#         roc4 = roc_auc_score(test_labels1, model4.predict(test4, verbose=0))
#         logger.info(
#             f"Model 4 evaluation: roc {round(roc4, 4)}, accuracy {round(acc4, 4)}, loss: {round(loss4, 4)}, mse {round(mse4, 4)}")
#         loss5, acc5, mse5 = model5.evaluate(test5, test_labels5, verbose=0)
#         roc5 = roc_auc_score(test_labels1, model5.predict(test5, verbose=0))
#         logger.info(
#             f"Model 5 evaluation: roc {round(roc5, 4)}, accuracy {round(acc5, 4)}, loss: {round(loss5, 4)}, mse {round(mse5, 4)}")
#
#
# def experiment_3_with_output():
#     data1, labels1 = load_client_data(num_parties=3, client_id=1, data=data_preproc, labels=data_labels)
#     data1 = data1[:, 1:]
#     train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 1 with shape {data1.shape}")
#
#     data2, labels2 = load_client_data(num_parties=3, client_id=2, data=data_preproc, labels=data_labels)
#     data2 = data2[:, 1:]
#     train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 2 with shape {data2.shape}")
#
#     data3, labels3 = load_client_data(num_parties=3, client_id=3, data=data_preproc, labels=data_labels)
#     data3 = data3[:, 1:]
#     train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)
#     logger.info(f"Splitting data for client 3 with shape {data3.shape}")
#
#     # Load simple model
#
#     model1 = load_simple_model_with_more_params(data1.shape[1], 'model1')
#     model2 = load_simple_model_with_more_params(data2.shape[1], 'model2')
#     model3 = load_simple_model_with_more_params(data3.shape[1], 'model3')
#
#     ensemble = load_ensemble([model1, model2, model3])
#
#     # Compile models
#     model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#
#     ensemble.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#     iterations_num = 2
#     for i in range(iterations_num):
#         logger.info(f'Iteration {str(i)} of {str(iterations_num)}')
#         # Train models
#         logger.info(f'Training model {model1.name}')
#         model1.fit(data1, labels1, epochs=4, batch_size=128, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model2.name}')
#         model2.fit(data2, labels2, epochs=4, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#         logger.info(f'Training model {model3.name}')
#         model3.fit(data3, labels3, epochs=1, batch_size=256, verbose=0, use_multiprocessing=True, workers=-1)
#
#         output1 = model1.predict(data1, verbose=0)
#         output2 = model2.predict(data2, verbose=0)
#         output3 = model3.predict(data3, verbose=0)
#
#         output1 = np.where(output1 > 0.5, 1, 0)
#         output2 = np.where(output2 > 0.5, 1, 0)
#         output3 = np.where(output3 > 0.5, 1, 0)
#
#         input_ensemble = np.column_stack((output1, output2, output3))
#         ensemble = load_base_of_ensemble(3, 'ensemble')
#         ensemble.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
#         ensemble.fit(input_ensemble, labels1, epochs=4, batch_size=128, verbose=0,
#                      use_multiprocessing=True, workers=-1)
#
#         output1_test = model1.predict(test1, verbose=0)
#         output2_test = model2.predict(test2, verbose=0)
#         output3_test = model3.predict(test3, verbose=0)
#
#         output1_test = np.where(output1_test > 0.5, 1, 0)
#         output2_test = np.where(output2_test > 0.5, 1, 0)
#         output3_test = np.where(output3_test > 0.5, 1, 0)
#
#         input_ensemble_test = np.column_stack((output1_test, output2_test, output3_test))
#         losse, acce, msee = ensemble.evaluate(input_ensemble_test, test_labels1, verbose=0)
#         predict_ensemble = ensemble.predict(input_ensemble_test, verbose=0)
#         roce = roc_auc_score(test_labels1, predict_ensemble)
#         logger.info(
#             f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")


def experiment_with_custom_hparams(used_dataset: str = None, model_type: str = None):
    # TODO Pensar en como hacer unas rondas de hiperparametizacion para cada modelo y luego entrenar el ensemble
    # Reflejar cuales son las caracteristicas menos importarntes

    data1, labels1 = load_client_data(num_parties=3, client_id=1, data=data_preproc, labels=data_labels)
    data1 = data1[:, 1:]
    train1, test1, train_labels1, test_labels1 = train_test_split(data1, labels1, test_size=0.2, random_state=42)
    logger.info(f"Splitting data for client 1 with shape {data1.shape}")

    data2, labels2 = load_client_data(num_parties=3, client_id=2, data=data_preproc, labels=data_labels)
    data2 = data2[:, 1:]
    train2, test2, train_labels2, test_labels2 = train_test_split(data2, labels2, test_size=0.2, random_state=42)
    logger.info(f"Splitting data for client 2 with shape {data2.shape}")

    data3, labels3 = load_client_data(num_parties=3, client_id=3, data=data_preproc, labels=data_labels)
    data3 = data3[:, 1:]
    train3, test3, train_labels3, test_labels3 = train_test_split(data3, labels3, test_size=0.2, random_state=42)
    logger.info(f"Splitting data for client 3 with shape {data3.shape}")

    # Load simple model

    model1 = load_simple_model(data1.shape[1], 'model1')
    model2 = load_simple_model(data2.shape[1], 'model2')
    model3 = load_simple_model(data3.shape[1], 'model3')

    ensemble = load_ensemble([model1, model2, model3])
    epoch1, epoch2, epoch3 = 0, 0, 0
    batch1, batch2, batch3 = 0, 0, 0
    # Optimizers
    if used_dataset == "adult_income":
        optimizer1 = RMSprop(learning_rate=0.02148694852725963)
        optimizer2 = RMSprop(learning_rate=0.03663001553939371)
        optimizer3 = RMSprop(learning_rate=0.0013639674172109756)
        epoch1, batch1 = 10, 64
        epoch2, batch2 = 4, 81
        epoch3, batch3 = 4, 46

    elif used_dataset == "breast_cancer":
        optimizer1 = RMSprop(learning_rate=0.022207703825774756)
        optimizer2 = RMSprop(learning_rate=0.04719563953394145)
        optimizer3 = RMSprop(learning_rate=0.009701722837890554)
        epoch1, batch1 = 7, 66
        epoch2, batch2 = 4, 37
        epoch3, batch3 = 3, 47
    else:
        raise ValueError("Dataset not specified")
    # Compile models
    model1.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
    model2.compile(optimizer=optimizer2, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
    model3.compile(optimizer=optimizer3, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)

    ensemble.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
    iterations_num = 2
    for i in range(iterations_num):
        logger.info(f'Iteration {str(i)} of {str(iterations_num)}')
        # Train models
        logger.info(f'Training model {model1.name}')
        model1.fit(data1, labels1, epochs=epoch1, batch_size=batch1, verbose=0, use_multiprocessing=True, workers=-1)
        logger.info(f'Training model {model2.name}')
        model2.fit(data2, labels2, epochs=epoch2, batch_size=batch2, verbose=0, use_multiprocessing=True, workers=-1)
        logger.info(f'Training model {model3.name}')
        model3.fit(data3, labels3, epochs=epoch3, batch_size=batch3, verbose=0, use_multiprocessing=True, workers=-1)

        logger.info('Evaluation of models after training')
        loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
        roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
        logger.info(
            f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
        loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
        roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
        logger.info(
            f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
        loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
        roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
        logger.info(
            f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")

        set_base_learner(ensemble, "model1", model1.get_weights())
        set_base_learner(ensemble, "model2", model2.get_weights())
        set_base_learner(ensemble, "model3", model3.get_weights())

        logger.info('Evaluation of ensemble before training')
        losse, acce, msee = ensemble.evaluate([test1, test2, test3], test_labels1, verbose=0)
        predict_ensemble = ensemble.predict([test1, test2, test3], verbose=0)
        roce = roc_auc_score(test_labels1, predict_ensemble)
        logger.info(
            f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")
        logger.info("Training ensemble")
        ensemble.fit([train1, train2, train3], train_labels1, epochs=4, batch_size=128, verbose=0,
                     use_multiprocessing=True, workers=-1)

        logger.info('Evaluation of ensemble after training')
        losse, acce, msee = ensemble.evaluate([test1, test2, test3], test_labels1, verbose=0)
        predict_ensemble = ensemble.predict([test1, test2, test3], verbose=0)
        roce = roc_auc_score(test_labels1, predict_ensemble)
        logger.info(
            f"Ensemble evaluation: roc {round(roce, 4)}, accuracy {round(acce, 4)}, loss: {round(losse, 4)}, mse {round(msee, 4)}")

        logger.info('Setting weights for base learners')
        model1 = load_simple_model(data1.shape[1], 'model1')
        model2 = load_simple_model(data2.shape[1], 'model2')
        model3 = load_simple_model(data3.shape[1], 'model3')

        model1.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
        model2.compile(optimizer=optimizer2, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)
        model3.compile(optimizer=optimizer3, loss='binary_crossentropy', metrics=['accuracy', 'mse'], jit_compile=True)

        bweights_model1 = model1.get_weights()
        model1.set_weights(get_base_learner(ensemble, "model1"))
        if not np.array_equal(bweights_model1, model1.get_weights()):
            logger.info(f'Weights are updated for model1')
        bweights_model2 = model2.get_weights()
        model2.set_weights(get_base_learner(ensemble, "model2"))
        if not np.array_equal(bweights_model2, model2.get_weights()):
            logger.info(f'Weights are updated for model2')
        bweights_model3 = model3.get_weights()
        model3.set_weights(get_base_learner(ensemble, "model3"))
        if not np.array_equal(bweights_model3, model3.get_weights()):
            logger.info(f'Weights are updated for model3')

        logger.info('Evaluation of models after setting weights and no training')
        loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)
        roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
        logger.info(
            f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
        loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
        roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
        logger.info(
            f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
        loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
        roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
        logger.info(
            f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")

        logger.info(f'Training model {model1.name}')
        model1.fit(data1, labels1, epochs=10, batch_size=64, verbose=0, use_multiprocessing=True, workers=-1)
        logger.info(f'Training model {model2.name}')
        model2.fit(data2, labels2, epochs=4, batch_size=81, verbose=0, use_multiprocessing=True, workers=-1)
        logger.info(f'Training model {model3.name}')
        model3.fit(data3, labels3, epochs=6, batch_size=49, verbose=0, use_multiprocessing=True, workers=-1)

        logger.info('Evaluation of models after training')
        loss1, acc1, mse1 = model1.evaluate(test1, test_labels1, verbose=0)

        roc1 = roc_auc_score(test_labels1, model1.predict(test1, verbose=0))
        logger.info(
            f"Model 1 evaluation: roc {round(roc1, 4)}, accuracy {round(acc1, 4)}, loss: {round(loss1, 4)}, mse {round(mse1, 4)}")
        loss2, acc2, mse2 = model2.evaluate(test2, test_labels2, verbose=0)
        roc2 = roc_auc_score(test_labels1, model2.predict(test2, verbose=0))
        logger.info(
            f"Model 2 evaluation: roc {round(roc2, 4)}, accuracy {round(acc2, 4)}, loss: {round(loss2, 4)}, mse {round(mse2, 4)}")
        loss3, acc3, mse3 = model3.evaluate(test3, test_labels3, verbose=0)
        roc3 = roc_auc_score(test_labels1, model3.predict(test3, verbose=0))
        logger.info(
            f"Model 3 evaluation: roc {round(roc3, 4)}, accuracy {round(acc3, 4)}, loss: {round(loss3, 4)}, mse {round(mse3, 4)}")


if __name__ == '__main__':
    data_preproc, data_labels = preprocess_data("adult_income")
    experiment_with_custom_hparams(used_dataset="breast_cancer", model_type="complejo")
