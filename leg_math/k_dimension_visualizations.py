import os
import numpy as np
import pandas as pd

import pickle

from leg_math.keras_helpers import NNnominate
from leg_math.data_processing import process_data

DATA_PATH = os.path.expanduser("~/data/leg_math/")

i = 1
dims = range(1, 11)
train_metrics_df = pd.DataFrame(index=dims, columns=["log_loss", "accuracy_score"])
test_metrics_df = pd.DataFrame(index=dims, columns=["log_loss", "accuracy_score"])

data_type = "cosponsor"
if data_type == "votes":
    vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
if data_type == "cosponsor":
    # vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data.feather")
    vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data_smart_oppose.feather")
    vote_df = vote_df.drop("icpsr_id", axis=1)
    sponsor_counts = vote_df.groupby("vote_id")["vote"].sum()
    min_sponsors = 1
    multi_sponsored_bills = sponsor_counts[sponsor_counts >= min_sponsors]
    multi_sponsored_bills.name = "sponsor_counts"
    vote_df = pd.merge(vote_df, multi_sponsored_bills.to_frame(), left_on="vote_id", right_index=True)
for i in range(1, 11):
    print(i)
    data_params = dict(
                   data_type=vote_df,
                   congress_cutoff=93,
                   k_dim=i,
                   k_time=1,
                   covariates_list=[],
                   )
    vote_data = process_data(**data_params, return_vote_df=False, unanimity_check=False)
    # vote_data, vote_df = process_data(**data_params, return_vote_df=True, unanimity_check=False)
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
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    fname_weights = '/models/' + data_type + '_model_weights_{congress_cutoff}_{k_dim}_{k_time}.h5'.format(**data_params)
    fname_history = '/models/' + data_type + '_train_history_{congress_cutoff}_{k_dim}_{k_time}.pkl'.format(**data_params)

    model.load_weights(DATA_PATH + fname_weights)
    fitted_model = model

    with open(DATA_PATH + fname_history, 'rb') as file_pi:
        history_dict = pickle.load(file_pi)

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

    train_metrics = model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
    train_metrics
    test_metrics = model.evaluate(x_test, vote_data["y_test"], batch_size=10000)
    test_metrics

    # train_metrics_df.loc[i, "log_loss"] = pd.Series(history_dict["loss"]).min()
    # train_metrics_df.loc[i, "accuracy_score"] = pd.Series(history_dict["acc"]).max()
    #
    # test_metrics_df.loc[i, "log_loss"] = pd.Series(history_dict["val_loss"]).min()
    # test_metrics_df.loc[i, "accuracy_score"] = pd.Series(history_dict["val_acc"]).max()
    train_metrics_df.loc[i, "log_loss"] = train_metrics[0]
    train_metrics_df.loc[i, "accuracy_score"] = train_metrics[1]

    test_metrics_df.loc[i, "log_loss"] = test_metrics[0]
    test_metrics_df.loc[i, "accuracy_score"] = test_metrics[1]

test_metrics_df["dataset"] = "test"

# TODO: rerun so I don't have to use this hack
# train_metrics_df["log_loss"] = train_metrics_df["log_loss"] / 2
# train_metrics_df["accuracy_score"] = train_metrics_df["accuracy_score"] + 0.2
train_metrics_df["dataset"] = "train"

metrics_df = pd.concat([test_metrics_df, train_metrics_df])
metrics_df.index.name = "k_dim"
metrics_df = metrics_df.reset_index()
metrics_df.to_pickle(DATA_PATH + data_type + "_{congress_cutoff}_metrics.pkl".format(**data_params))

plot_data = metrics_df.set_index(["k_dim", "dataset"]).unstack(level=["dataset"]).unstack().reset_index()
plot_data = plot_data.rename(columns={"level_0": "metric", 0: "score"})


# losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['loss']))],
#                        'training': [loss for loss in history_dict['loss']],
#                        'validation': [loss for loss in history_dict['val_loss']],
#                        })
# ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
# ax.set_ylabel("log loss")
# ax.set_ylim([0.0, 3.0])
#
# losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['acc']))],
#                        'training': [loss for loss in history_dict['acc']],
#                        'validation': [loss for loss in history_dict['val_acc']],
#                        })
# ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
# ax.set_ylabel("log loss")
# ax.set_ylim([0.0, 3.0])
