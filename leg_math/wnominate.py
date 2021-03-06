import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from tensorflow.keras.losses import BinaryCrossentropy

from leg_math.keras_helpers import GetBest, NNnominate
from leg_math.data_processing import process_data

# Data processing
from constants import DATA_PATH

i = 3
for i in range(2, 6):
    data_params = dict(
                   data_type="test",
                   congress_cutoff=0,
                   k_dim=i,
                   k_time=0,
                   covariates_list=["in_majority"],
                   )
    if data_params["data_type"] == "cosponsor":
        vote_data = process_data(**data_params, return_vote_df=False, unanimity_check=False)
    else:
        vote_data = process_data(**data_params, return_vote_df=False, unanimity_check=True)
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
    bce = BinaryCrossentropy()
    model.compile(loss=bce, optimizer='Nadam', metrics=['accuracy'])

    sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(vote_data["y_train"]))

    callbacks = [EarlyStopping('val_loss', patience=20, restore_best_weights=True),
                 # GetBest(monitor='val_loss', verbose=1, mode='auto'),
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
                        validation_data=(x_test, vote_data["y_test"]), verbose=1, callbacks=callbacks,
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
    model.layers
    model.get_layer("main_output").get_weights()[0]
    np.log(model.get_layer("wnom_term").get_weights()[0])
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


model.get_layer("main_output").get_weights()[0]
model.get_layer("wnom_term").get_weights()[0].round(5)
(~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)).sum()

k_dim = model_params["k_dim"]
k_time = model_params["k_time"]
ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
drift_names = ["drift_{}".format(j) for j in range(1, k_time + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]

if data_params["data_type"] == "test":
    var_list = [f"nominate_dim{i}" for i in range(1, k_dim + 1)] + ideal_point_names + drift_names
if data_params["data_type"] == "votes" or data_params["data_type"] == "cosponsor":
    var_list = [f"nominate_dim{i}" for i in range(1, 3)] + ideal_point_names + drift_names

cf_ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])

cf_ideal_points.corr()
import seaborn as sns
sns.set(style="white")

g = sns.PairGrid(cf_ideal_points, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors="C0")
g.map_diag(sns.kdeplot, lw=2)

random_votes = pd.read_feather(DATA_PATH + "/test_votes_df.feather")

leg_test = pd.Series(x_test[0], name="leg_id").map(vote_data["leg_crosswalk"])
vote_test = pd.Series(x_test[1], name="bill_id").map(vote_data["vote_crosswalk"])
test_df = pd.concat([leg_test, vote_test], axis=1)
test_df["pred_proba"] = model.predict(x_test, batch_size=10000)

asdf = pd.merge(test_df, random_votes[["leg_id", "bill_id", "vote_prob"]], on=["leg_id", "bill_id"])
asdf.corr()
asdf.plot(kind='scatter', x="vote_prob", y="pred_proba", alpha=0.1)
sns.regplot(x="vote_prob", y="pred_proba", data=asdf, lowess=True, scatter_kws={'alpha': 0.1}, line_kws={'color': "red"})
leg_stuff = pd.merge(cf_ideal_points, random_votes[["leg_id", "coord1D", "coord2D", "coord3D"]].drop_duplicates(), left_index=True, right_on="leg_id")

g = sns.PairGrid(leg_stuff, diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot, colors="C0")
g.map_diag(sns.kdeplot, lw=2)
leg_stuff.corr()
