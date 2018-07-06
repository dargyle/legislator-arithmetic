import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.callbacks import EarlyStopping, TerminateOnNaN

from leg_math.keras_helpers import GetBest, NNnominate
from leg_math.data_processing import process_data

# Data processing
DATA_PATH = os.path.expanduser("~/data/leg_math/")

i=1
for i in range(1, 9):
    data_params = dict(
                   data_type="votes",
                   congress_cutoff=93,
                   k_dim=i,
                   k_time=1,
                   covariates_list=["in_majority"],
                   )

    vote_data = process_data(**data_params, return_vote_df=False)
    # vote_data, vote_df = process_data(**data_params, return_vote_df=True)
    model_params = {
                    "n_leg": vote_data["J"],
                    "n_votes": vote_data["M"],
                    "k_dim": data_params["k_dim"],
                    "k_time": data_params["k_time"],
                    "init_leg_embedding": vote_data["init_embedding"],
                    "yes_point_dropout": 0.0,
                    "no_point_dropout": 0.0,
                    "combined_dropout": 0.5,
                    "dropout_type": "timestep",
                    "covariates_list": data_params["covariates_list"],
                    }

    print("N Legislators: {}".format(vote_data["J"]))
    print("N Votes: {}".format(vote_data["M"]))
    print("N: {}".format(vote_data["N"]))

    print("Let's do a keras cf example")
    print(vote_data["y_train"].mean())

    model = NNnominate(**model_params)

    model.summary()

    SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # model.compile(loss='mse', optimizer='adamax')
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(vote_data["y_train"]))

    callbacks = [EarlyStopping('val_loss', patience=50),
                 GetBest(monitor='val_loss', verbose=1, mode='auto'),
                 TerminateOnNaN()]
    if data_params["covariates_list"]:
        if data_params["k_time"] > 0:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + vote_data["time_passed_train"] + [vote_data["covariates_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + vote_data["time_passed_test"]+ [vote_data["covariates_test"]]
        else:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + [vote_data["covariates_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + [vote_data["covariates_test"]]
    else:
        if data_params["k_time"] > 0:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + vote_data["time_passed_train"]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + vote_data["time_passed_test"]
        else:
            x_train = [vote_data["j_train"], vote_data["m_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]]
    history = model.fit(x_train, vote_data["y_train"], epochs=5000, batch_size=32768,
                        validation_data=(x_test, vote_data["y_test"]), verbose=2, callbacks=callbacks,
                        class_weight={0: sample_weights[0],
                                      1: sample_weights[1]})

    # TODO: Calculate the relevant metrics and optionally save
    params_and_results = data_params.copy()
    params_and_results.update(model_params)
    params_and_results.pop("init_leg_embedding")

    train_metrics = model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
    params_and_results["loss"] = train_metrics[0]
    params_and_results["acc"] = train_metrics[1]
    test_metrics = model.evaluate(x_test, vote_data["y_test"], batch_size=10000)
    params_and_results["val_loss"] = test_metrics[0]
    params_and_results["val_acc"] = test_metrics[1]

    # Count the number of nonzero dimension salience weights
    valid_weight_count = ~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)
    params_and_results["nonzero_dim_count"] = valid_weight_count.sum()

    params_and_results = pd.Series(params_and_results)
    print(params_and_results)

    experimental_log_path = os.getcwd() + f'/experimental_log/{data_params["data_type"]}_model_metrics.csv'

    try:
        exp_log = pd.read_csv(experimental_log_path, index_col=False)
        exp_log = exp_log.append(params_and_results.to_frame().transpose())
    except IOError:
        exp_log = params_and_results.to_frame().transpose()

    exp_log.to_csv(experimental_log_path, index=False)

    # model.save(DATA_PATH + '/models/{data_type}_model_{congress_cutoff}_{k_dim}_{k_time}.h5'.format(**data_params))
    fname_weights = '/models/{data_type}_model_weights_{congress_cutoff}_{k_dim}_{k_time}.h5'.format(**data_params)
    model.save_weights(DATA_PATH + fname_weights)

    fname_history = '/models/{data_type}_train_history_{congress_cutoff}_{k_dim}_{k_time}.pkl'.format(**data_params)
    with open(DATA_PATH + fname_history, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    history_dict = history.history
