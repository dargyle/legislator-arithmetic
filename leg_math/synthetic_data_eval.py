import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from data_generation.data_processing import process_data, prep_r_rollcall
from data_generation.random_votes import generate_nominate_votes

from leg_math.keras_helpers import GetBest, NNnominate

from scipy import stats

from sklearn.metrics import log_loss, accuracy_score

from constants import DATA_PATH

# R imports for R integration
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

base = importr("base")
pscl = importr("pscl")
wnominate = importr("wnominate")

SYNTHETIC_PATH = DATA_PATH + '/synthetic/'


def get_probs_from_nominate(votes, ideal_points, yes_points, no_points, w, beta, cdf_type="norm"):
    '''Predict probabilities using the parameters estimated from a nominate style model

    # Args:
        votes (DataFrame): a list of votes to estimate probabilities, at a minimum requires a leg_id
            vote_id, and a vote column
        ideal_points (DataFrame): The estimated ideal points from a model, indexed by leg_id
        yes_points (DataFrame): The estimated yes points from the a model, indexed by vote_id
        no_points (DataFrame): The estimated no points from the a model, indexed by vote_id
        w (np.arry): A vector of length d, where d is the number of dimensions in the model,
            containing the estimated dimension salience weights
        beta (float): The estimated model parameter on the nominate term
        cdf_type (str): The cdf function to use, supported values are "norm" and "logit"
    # Returns:
        temp_data (DataFrame): A data frame of predicted probabilities
    '''
    temp_data = pd.merge(votes, ideal_points, left_on="leg_id", right_index=True)
    temp_data = pd.merge(temp_data, yes_points, left_on="vote_id", right_index=True)
    temp_data = pd.merge(temp_data, no_points, left_on="vote_id", right_index=True)

    if temp_data.isnull().any().any():
        n = temp_data.shape
        temp_data = temp_data.dropna().copy()
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


# Load synthetic votes
use_cached_votes = True
if use_cached_votes:
    vote_df = pd.read_feather(SYNTHETIC_PATH + "/test_votes_df_norm.feather")
else:
    random_votes = generate_nominate_votes(w=np.array([1.5, 0.75, 0.75]))
    random_votes = random_votes.reset_index()
    vote_df = random_votes.copy()
# if data_type == "votes":
#     vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
#     # Limit to senate for now
#     vote_df = vote_df[vote_df["chamber"] == "Senate"]

# print("Get wnominate estimate")
# Old way: run an R script
# robjects.r['source']("./leg_math/test_data_in_r.R")
# Better way, directly interact with R (below)

i = 3
top_dim = 7

metrics_list = []
for i in range(1, top_dim):
    # wnom = robjects.r[f"wnom{i}"]
    # wnom.rx2("legislators")

    print("Get nn_estimate")
    # Process the vote_df into the model data format
    data_params = dict(
                   vote_df=vote_df,
                   congress_cutoff=112,
                   k_dim=i,
                   k_time=0,
                   covariates_list=[],
                   )
    vote_data = process_data(**data_params)

    # Cache some objects for use in R later
    roll_call = prep_r_rollcall(vote_data)

    leg_info_cols = ["leg_id", "partyCode", "icpsrState"] + [f"coord{i}D" for i in range(1, 4)]
    leg_data = vote_df[leg_info_cols].drop_duplicates()

    vote_info_cols = ["vote_id", "congress"] + [f"yes_coord{i}D" for i in range(1, 4)] + [f"no_coord{i}D" for i in range(1, 4)]
    vote_metadata = vote_df[vote_info_cols].drop_duplicates()

    model_params = {
                    "n_leg": vote_data["J"],
                    "n_votes": vote_data["M"],
                    "k_dim": data_params["k_dim"],
                    "k_time": data_params["k_time"],
                    "init_leg_embedding": vote_data["init_embedding"],
                    "yes_point_dropout": 0.0,
                    "no_point_dropout": 0.0,
                    "combined_dropout": 0.25,
                    "dropout_type": "timestep",
                    "covariates_list": data_params["covariates_list"],
                    }

    model = NNnominate(**model_params)

    # model.summary()
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))

    # model.compile(loss='mse', optimizer='adamax')
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

    # Weights are probably better, but not in original so comment out here
    weight_by_frequency = False
    if weight_by_frequency:
        sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(vote_data["y_train"]))
    else:
        sample_weights = {k: 1 for k in np.unique(vote_data["y_train"])}

    callbacks = [EarlyStopping('val_loss', patience=20, restore_best_weights=True),
                 # GetBest(monitor='val_loss', verbose=1, mode='auto'),
                 TerminateOnNaN()]
    if data_params["covariates_list"]:
        if data_params["k_time"] > 0:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + vote_data["time_passed_train"] + [vote_data["covariates_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + vote_data["time_passed_test"] + [vote_data["covariates_test"]]
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

    train_df = np.stack([vote_data["j_train"],
                         vote_data["m_train"],
                         vote_data["y_train"]], axis=1)
    train_df = pd.DataFrame(train_df, columns=["leg_id", "vote_id", "vote"])
    train_df["leg_id"] = train_df["leg_id"].map(vote_data["leg_crosswalk"])
    train_df["vote_id"] = train_df["vote_id"].map(vote_data["vote_crosswalk"])
    test_df = np.stack([vote_data["j_test"],
                        vote_data["m_test"],
                        vote_data["y_test"]], axis=1)
    test_df = pd.DataFrame(test_df, columns=["leg_id", "vote_id", "vote"])
    test_df["leg_id"] = test_df["leg_id"].map(vote_data["leg_crosswalk"])
    test_df["vote_id"] = test_df["vote_id"].map(vote_data["vote_crosswalk"])

    train_metrics = model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
    train_metrics
    test_metrics = model.evaluate(x_test, vote_data["y_test"], batch_size=10000)
    test_metrics

    k_dim = i
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

    # roll_call = train_df.copy()
    # # roll_call.set_index(["leg_id", "vote_id"])["vote"].unstack().notnull().sum(axis=0)
    # roll_call["vote"] = roll_call["vote"].map({1: 1, 0: 6})
    # roll_call = roll_call.set_index(["leg_id", "vote_id"])["vote"].unstack()
    # roll_call = roll_call.fillna(9).astype(int)
    # # roll_call.to_csv(DATA_PATH + "/test_votes.csv")
    # # leg_data.to_csv(DATA_PATH + "/test_legislators.csv")
    # (roll_call.iloc[:, -5:] != 9).describe()

    roll_call_mat = base.as_matrix(roll_call)
    # roll_call_mat = base.as_matrix(pandas2ri.py2ri(roll_call))

    # if data_type == "test":
    #     leg_data = pd.read_csv("~/data/leg_math/test_legislators.csv")

    leg_data.columns = ["true_" + str(col) if "coord" in col else col for col in leg_data.columns]

    # if data_type == "votes":
    #     leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")
    #     leg_data = leg_data[["leg_id", "state_abbrev", "bioname", "party_code"]].drop_duplicates().set_index("leg_id")
    #     # leg_data = leg_data.[leg_data["leg_id"].isin(train_df["leg_id"].values)]
    #     leg_data = leg_data.loc[roll_call.index.values, :]

    r_data = pscl.rollcall(roll_call_mat, yea=1, nay=6, notInLegis=9, missing=np.nan,
                           legis_names=roll_call.index,
                           legis_data=leg_data,
                           # vote_names=roll_call.columns,
                           # vote_data=pd.DataFrame(roll_call.columns, columns=["vote_id"]),
                           )
    wnom = wnominate.wnominate(r_data, polarity=base.seq(1, i), dims=i)
    base.summary(wnom)

    k_dim = data_params["k_dim"]
    k_time = data_params["k_time"]
    # if data_params["data_type"] == "test":
    #     test_actual = False
    #     if test_actual:
    #         leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0).reset_index()
    #     else:
    #         leg_data = pd.read_csv(DATA_PATH + f"/wnom{k_dim}D_results.csv", index_col=0)
    #         leg_data = pandas2ri.ri2py(wnom.rx2("legislators"))
    #         leg_data.index.name = "leg_id"
    #         leg_data = leg_data.reset_index()
    leg_data

    wnom_votes = wnom.rx2("rollcalls")
    base.rownames(wnom.rx2("rollcalls"))
    wnom_votes.index = roll_call.columns
    wnom_votes[wnom_votes.isnull().any(axis=1)]

    for i in range(1, k_dim + 1):
        wnom_votes[f"yes_coord{i}D"] = wnom_votes[f"midpoint{i}D"] - (wnom_votes[f"spread{i}D"] / 2)
        wnom_votes[f"no_coord{i}D"] = wnom_votes[f"midpoint{i}D"] + (wnom_votes[f"spread{i}D"] / 2)

    r_ideal_points = wnom.rx2("legislators")
    # r_ideal_points.groupby("party_code")["coord1D"].agg(["mean", "std"])
    # r_ideal_points = leg_data.set_index("leg_id")[[f"coord{i}D" for i in range(1, k_dim + 1)]]
    # r_ideal_points.columns = [f"coord{i}D" for i in range(1, k_dim + 1)]

    # votes = train_df
    train_results = get_probs_from_nominate(train_df,
                                            ideal_points=r_ideal_points,
                                            yes_points=wnom_votes.filter(regex="yes_coord"),
                                            no_points=wnom_votes.filter(regex="no_coord"),
                                            w=np.array(wnom.rx2("weights")),
                                            beta=np.array(wnom.rx2("beta")),
                                            cdf_type="norm")

    test_results = get_probs_from_nominate(test_df,
                                           ideal_points=r_ideal_points,
                                           yes_points=wnom_votes.filter(regex="yes_coord"),
                                           no_points=wnom_votes.filter(regex="no_coord"),
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

    combined_ideal = pd.merge(ideal_points, r_ideal_points,
                              left_index=True, right_index=True,
                              suffixes=("_nn", "_wnom"))
    combined_ideal.to_pickle(SYNTHETIC_PATH + f"synthetic_{k_dim}_combined_ideal_data.pkl")
    # sns.pairplot(combined_ideal.filter(regex=r"coord\dD_"), size=3, diag_kind="kde", plot_kws={'alpha': 0.5})
    # def corrfunc(x, y, **kws):
    #     r, _ = stats.pearsonr(x, y)
    #     ax = plt.gca()
    #     ax.annotate("r = {:.2f}".format(r), xy=(.9, .1), xycoords=ax.transAxes)
    # g = sns.pairplot(combined_ideal.filter(regex=r"coord\dD_nn|true_coord"), size=3, diag_kind="kde", plot_kws={'alpha': 0.5}); g.map_lower(corrfunc)

final_metrics = pd.concat(metrics_list)
final_metrics.to_pickle(DATA_PATH + f"test_data_metrics.pkl")

final_metrics = pd.read_pickle(DATA_PATH + f"test_data_metrics.pkl")
print(final_metrics.set_index(["k_dim", "algorithm", "dataset"]).unstack(level=["algorithm", "dataset"]))

import seaborn as sns
import matplotlib.pyplot as plt

# plot_data = final_metrics[final_metrics["algorithm"] == "nn"].set_index(["k_dim", "dataset"])["log_loss"].unstack("dataset").reset_index()
# plot_data = final_metrics.set_index(["k_dim", "dataset"])["log_loss"].reset_index()

final_metrics = pd.read_pickle(DATA_PATH + f"test_data_metrics.pkl")
plot_data = final_metrics.set_index(["k_dim", "algorithm", "dataset"]).unstack(level=["algorithm", "dataset"]).unstack().reset_index()
plot_data = plot_data.rename(columns={"level_0": "metric", 0: "score"})
plot_data["Model"] = plot_data["algorithm"] + "_" + plot_data["dataset"]
plot_data["Model"] = plot_data["Model"].map({"nn_train": "NN-NOMINATE (train)",
                                             "nn_test": ("NN-NOMINATE (test)"),
                                             "wnominate_train": "WNOMINATE (train)",
                                             "wnominate_test": "WNOMINATE (test)"})
# plot_data = plot_data[plot_data["metric"] == "log_loss"]

ax = sns.lineplot(data=plot_data[(plot_data["metric"] == "log_loss")],
                  x="k_dim", y="score", estimator=None, hue="Model", marker="o")
ax = sns.lineplot(data=plot_data[(plot_data["metric"] == "log_loss") & (plot_data["algorithm"] == "wnominate")],
                  x="k_dim", y="score", estimator=None, hue="Model", marker="^")
ax = sns.lineplot(data=plot_data[(plot_data["metric"] == "log_loss") & (plot_data["algorithm"] == "nn")],
                  x="k_dim", y="score", estimator=None, hue="Model", marker="^")
ax.set(xlim=(0.9, 5.1), xticks=range(1, top_dim))
fig = ax.get_figure()
fig.show()
# fig.savefig(, bbox_inches='tight')
fig.clf()

plt.figure(1)
plt.subplot(211)
ax = sns.tsplot(data=plot_data[plot_data["metric"] == "log_loss"], time="k_dim", unit="algorithm", condition="dataset", value="score",
                err_style=None, marker="o")
ax.set(xlim=(0.9, 5.1), xticks=range(1, 8))
fig = ax.get_figure()
fig.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# sns.regplot(x, y, ax=ax1)
# sns.kdeplot(x, ax=ax2)
sns.tsplot(data=plot_data[(plot_data["metric"] == "log_loss")],
           time="k_dim", unit="Model", condition="Model", value="score",
           err_style=None, marker="o", ax=ax1)
ax1.lines[-1].set_marker("^")
ax1.lines[-2].set_marker("^")
ax1.legend()
ax1.set(xlim=(0.9, top_dim - 1 + 0.1), xticks=range(1, top_dim), ylabel="log-loss", xlabel="Number of Dimensions")
sns.tsplot(data=plot_data[(plot_data["metric"] == "accuracy_score")],
           time="k_dim", unit="Model", condition="Model", value="score",
           err_style=None, marker="o", ax=ax2)
ax2.lines[-1].set_marker("^")
ax2.lines[-2].set_marker("^")
ax2.legend()
ax2.set(xlim=(0.9, top_dim - 1 + 0.1), xticks=range(1, top_dim), ylabel="accuracy", xlabel="Number of Dimensions")
fig

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes



combined_ideal = pd.read_pickle(DATA_PATH + "votes_2_combined_ideal_data.pkl")

import pytablewriter
writer = pytablewriter.MarkdownTableWriter()
writer.from_dataframe(combined_ideal.filter(regex=r"coord\dD_").corr().reset_index())
writer.write_table()
