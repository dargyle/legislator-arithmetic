import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.callbacks import EarlyStopping, TerminateOnNaN

from leg_math.keras_helpers import GetBest, NNnominate
from leg_math.data_processing import process_data
from leg_math.random_votes import generate_random_votes

from scipy import stats

from sklearn.metrics import log_loss, accuracy_score

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

DATA_PATH = os.path.expanduser("~/data/leg_math/")


def get_probs_from_nominate(votes, ideal_points, yes_points, no_points, w, beta, cdf_type="logit"):
    temp_data = pd.merge(votes, ideal_points, left_on="leg_id", right_index=True)
    temp_data = pd.merge(temp_data, yes_points, left_on="vote_id", right_index=True)
    temp_data = pd.merge(temp_data, no_points, left_on="vote_id", right_index=True)

    if temp_data.isnull().any().any():
        n = temp_data.shape
        temp_data = temp_data.dropna()
        print(f"Dropped {n[0] - temp_data.shape[0]} votes with missing values")

    for i in range(1, k_dim + 1):
        temp_data[f"yes_diff_{i}D"] = (temp_data[f"coord{i}D"] - temp_data[f"yes_coord{i}D"]) ** 2
        temp_data[f"no_diff_{i}D"] = (temp_data[f"coord{i}D"] - temp_data[f"no_coord{i}D"]) ** 2

    temp_data["yes_sum"] = np.dot(temp_data.filter(regex="yes_diff"), w.transpose() ** 2)
    temp_data["no_sum"] = np.dot(temp_data.filter(regex="no_diff"), w.transpose() ** 2)

    temp_data["wnom_sum"] = np.exp(-0.5 * temp_data["yes_sum"]) - np.exp(-0.5 * temp_data["no_sum"])
    # stats.norm.cdf(beta * wnom_sum)
    if cdf_type == "logit":
        temp_data["vote_prob"] = stats.logistic.cdf(beta[0] * temp_data["wnom_sum"])
    elif cdf_type == "norm":
        temp_data["vote_prob"] = stats.norm.cdf(beta[0] * temp_data["wnom_sum"])
    temp_data[temp_data.isnull().any(axis=1)].transpose()

    assert not temp_data.isnull().any().any(), "missing values in temp_data"

    return temp_data[["leg_id", "vote_id", "vote", "vote_prob"]]

# Call generate random votes to ensure data exists
_ = generate_random_votes()


print("Get wnominate estimate")
robjects.r['source']("./leg_math/test_data_in_r.R")

i = 5

metrics_list = []
for i in range(1, 6):
    wnom = robjects.r[f"wnom{i}"]
    wnom.rx2("legislators")

    print("Get nn_estimate")
    data_params = dict(
                   data_type="test",
                   congress_cutoff=114,
                   k_dim=i,
                   k_time=0,
                   covariates_list=[],
                   )
    # vote_data = process_data(**data_params, return_vote_df=False)
    vote_data, vote_df = process_data(**data_params, return_vote_df=True)
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

    k_dim = data_params["k_dim"]
    k_time = data_params["k_time"]
    if data_params["data_type"] == "test":
        test_actual = False
        if test_actual:
            leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0).reset_index()
        else:
            leg_data = pd.read_csv(DATA_PATH + f"/wnom{k_dim}D_results.csv", index_col=0)
            leg_data = pandas2ri.ri2py(wnom.rx2("legislators"))
            leg_data.index.name = "leg_id"
            leg_data = leg_data.reset_index()

        # leg_data = leg_data.rename(columns={"index": "leg_id",
        #                                     "icpsrState": "state_icpsr",
        #                                     "coord1D": "nominate_dim1",
        #                                     "coord2D": "nominate_dim2",
        #                                     "coord3D": "nominate_dim3",
        #                                     "coord4D": "nominate_dim4",
        #                                     "coord4D": "nominate_dim4",
        #                                     "coord5D": "nominate_dim5",
        #                                     "party.1": "party_code",
        #                                     "partyCode": "party_code"})
        # leg_data["bioname"] = ""
        # leg_data
        #
        # # fitted_model.get_layer("main_output").get_weights()[0]
        # # fitted_model.get_layer("yes_term").get_weights()[0]
        # # fitted_model.get_layer("no_term").get_weights()[0]
        # # fitted_model.get_layer("yes_point").get_weights()[0]
        # # np.exp(0.5)
        # col_names = (["leg_id", "state_icpsr", "bioname", "party_code"] +
        #              [f"nominate_dim{i}" for i in range(1, k_dim + 1)] +
        #              [f"coord{i}D" for i in range(1, k_dim + 1)])

    train_metrics = model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
    train_metrics
    test_metrics = model.evaluate(x_test, vote_data["y_test"], batch_size=10000)
    test_metrics

    votes = vote_df[["leg_id", "vote_id", "vote"]].copy()

    votes = votes.sample(frac=1, replace=False, random_state=42)
    N = len(vote_df)
    key_index = round((1 - 0.2) * N)

    train_df = votes.iloc[:key_index, :]
    test_df = votes.iloc[key_index:, :]

    ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0],
                                columns=[f"coord{i}D" for i in range(1, k_dim + 1)])
    ideal_points.index = ideal_points.index.map(vote_data["leg_crosswalk"])
    yes_points = pd.DataFrame(model.get_layer("yes_point").get_weights()[0],
                              columns=[f"yes_coord{i}D" for i in range(1, k_dim +1)])
    yes_points.index = yes_points.index.map(vote_data["vote_crosswalk"])
    no_points = pd.DataFrame(model.get_layer("no_point").get_weights()[0],
                             columns=[f"no_coord{i}D" for i in range(1, k_dim +1)])
    no_points.index = no_points.index.map(vote_data["vote_crosswalk"])
    w = model.get_layer("wnom_term").get_weights()[0]
    beta = model.get_layer("main_output").get_weights()[0]

    train_results = get_probs_from_nominate(train_df, ideal_points, yes_points, no_points, w, beta)
    nn_train_metrics = pd.Series({"k_dim": i,
                                  "dataset": "train",
                                  "algorithm": "nn",
                                  "log_loss": log_loss(train_results["vote"], train_results["vote_prob"]),
                                  "accuracy_score": accuracy_score(train_results["vote"], 1 * (train_results["vote_prob"] > 0.5))})
    test_results = get_probs_from_nominate(test_df, ideal_points, yes_points, no_points, w, beta)
    nn_test_metrics = pd.Series({"k_dim": i,
                                 "dataset": "test",
                                 "algorithm": "nn",
                                 "log_loss": log_loss(test_results["vote"], test_results["vote_prob"]),
                                 "accuracy_score": accuracy_score(test_results["vote"], 1 * (test_results["vote_prob"] > 0.5))})

    roll_calls = pd.read_csv(DATA_PATH + "wnom3D_rollcalls.csv", index_col=0)
    roll_calls = pandas2ri.ri2py(wnom.rx2("rollcalls"))
    for i in range(1, k_dim + 1):
        roll_calls[f"yes_coord{i}D"] = roll_calls[f"midpoint{i}D"] - roll_calls[f"spread{i}D"] / 2
        roll_calls[f"no_coord{i}D"] = roll_calls[f"midpoint{i}D"] + roll_calls[f"spread{i}D"] / 2

    r_ideal_points = leg_data.set_index("leg_id")[[f"coord{i}D" for i in range(1, k_dim + 1)]]
    # r_ideal_points.columns = [f"coord{i}D" for i in range(1, k_dim + 1)]

    train_results = get_probs_from_nominate(train_df,
                                            ideal_points=r_ideal_points,
                                            yes_points=roll_calls.filter(regex="yes_coord"),
                                            no_points=roll_calls.filter(regex="no_coord"),
                                            w=np.array(wnom.rx2("weights")),
                                            beta=np.array(wnom.rx2("beta")),
                                            cdf_type="norm")

    test_results = get_probs_from_nominate(test_df,
                                           ideal_points=r_ideal_points,
                                           yes_points=roll_calls.filter(regex="yes_coord"),
                                           no_points=roll_calls.filter(regex="no_coord"),
                                           w=np.array(wnom.rx2("weights")),
                                           beta=np.array(wnom.rx2("beta")),
                                           cdf_type="norm")

    wnominate_train_metrics = pd.Series({"k_dim": i,
                                         "dataset": "train",
                                         "algorithm": "wnominate",
                                         "log_loss": log_loss(train_results["vote"], train_results["vote_prob"]),
                                         "accuracy_score": accuracy_score(train_results["vote"], 1 * (train_results["vote_prob"] > 0.5))})
    wnominate_test_metrics = pd.Series({"k_dim": i,
                                        "dataset": "test",
                                        "algorithm": "wnominate",
                                        "log_loss": log_loss(test_results["vote"], test_results["vote_prob"]),
                                        "accuracy_score": accuracy_score(test_results["vote"], 1 * (test_results["vote_prob"] > 0.5))})

    metrics_list += [pd.concat([nn_train_metrics,
                                nn_test_metrics,
                                wnominate_train_metrics,
                                wnominate_test_metrics], axis=1).transpose()]

final_metrics = pd.concat(metrics_list)
final_metrics.to_pickle(DATA_PATH + "test_data_metrics.pkl")

final_metrics = pd.read_pickle(DATA_PATH + "test_data_metrics.pkl")
final_metrics.set_index(["k_dim", "algorithm", "dataset"]).unstack(level=["algorithm", "dataset"])
