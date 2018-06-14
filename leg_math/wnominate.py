import os
import numpy as np
import pandas as pd

import pickle

import warnings

from keras.layers import Embedding, Reshape, Merge, Dropout, SpatialDropout1D, Dense, Flatten, Input, Dot, LSTM, Add, Subtract, Conv1D, MaxPooling1D, Concatenate, Multiply, BatchNormalization, Lambda, Activation
from keras.models import Sequential, Model
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras.regularizers import Regularizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.constraints import Constraint, unit_norm, MinMaxNorm
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras import optimizers
from keras.utils.generic_utils import get_custom_objects
from keras.layers import constraints

from keras.models import load_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras import backend as K

from leg_math.keras_helpers import GetBest, MOAmodels

# Data processing
DATA_PATH = os.path.expanduser("~/data/leg_math/")

data_type = "test"
if data_type == "votes":
    vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
if data_type == "cosponsor":
    # vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data.feather")
    vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data_smart_oppose.feather")
    sponsor_counts = vote_df.groupby("vote_id")["vote"].sum()
    min_sponsors = 2
    multi_sponsored_bills = sponsor_counts[sponsor_counts >= min_sponsors]
    multi_sponsored_bills.name = "sponsor_counts"
    vote_df = pd.merge(vote_df, multi_sponsored_bills.to_frame(), left_on="vote_id", right_index=True)
if data_type == "test":
    roll_call_object = pd.read_csv(DATA_PATH + "/test_votes.csv", index_col=0)
    vote_df = roll_call_object.replace({1: 1,
                                        2: 1,
                                        3: 1,
                                        4: 0,
                                        5: 0,
                                        6: 0,
                                        7: np.nan,
                                        8: np.nan,
                                        9: np.nan,
                                        0: np.nan})
    vote_df = vote_df.stack().reset_index()
    assert not vote_df.isnull().any().any(), "mising codes in votes"
    vote_df.columns = ["leg_id",  "vote_id", "vote"]
    vote_df["congress"] = 115
    vote_df["chamber"] = "s"
    leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0)
    if "partyCode" in leg_data.columns:
        leg_data["init_value"] = leg_data["partyCode"].map({100: -1,
                                                            200: 1})
    else:
        leg_data["init_value"] = leg_data["party.1"].map({100: -1,
                                                          200: 1})

    # leg_data["init_value"] = 1
    vote_df = pd.merge(vote_df, leg_data[["init_value"]], left_on="leg_id", right_index=True)

congress_cutoff = 0
if congress_cutoff:
    vote_df = vote_df[vote_df["congress"] >= congress_cutoff]

first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
first_session.columns = ["first_session", "last_session"]
# first_session["first_session"].value_counts()
vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]

leg_ids = vote_df["leg_id"].unique()
vote_ids = vote_df["vote_id"].unique()

vote_df_temp = vote_df.copy()
leg_crosswalk = pd.Series(leg_ids).to_dict()
leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
vote_crosswalk = pd.Series(vote_ids).to_dict()
vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

vote_df_temp["leg_id"] = vote_df_temp["leg_id"].map(leg_crosswalk_rev)
vote_df_temp["vote_id"] = vote_df_temp["vote_id"].map(vote_crosswalk_rev)
# Shuffle the order of the vote data
vote_df_temp = vote_df_temp.sample(frac=1, replace=False)

init_embedding = vote_df_temp[["leg_id", "init_value"]].drop_duplicates("leg_id").set_index("leg_id").sort_index()

k_time = 0

vote_data = {'J': len(leg_ids),
             'M': len(vote_ids),
             'N': len(vote_df_temp),
             'j': vote_df_temp["leg_id"].values,
             'm': vote_df_temp["vote_id"].values,
             'y': vote_df_temp["vote"].astype(int).values,
             'time_passed': [(vote_df_temp["time_passed"] ** i).values for i in range(1, k_time + 1)],
             'init_embedding': init_embedding,
             'vote_crosswalk': vote_crosswalk,
             'leg_crosswalk': leg_crosswalk}

print("N Legislators: {}".format(vote_data["J"]))
print("N Votes: {}".format(vote_data["M"]))
print("N: {}".format(vote_data["N"]))

print("Let's do a keras cf example")
print(vote_data["y"].mean())

n_leg = vote_data["J"]
n_votes = vote_data["M"]
k_dim = 4

# use_popularity = True
ideal_dropout = 0.0
yes_point_dropout = 0.0
no_point_dropout = 0.0
leg_input_dropout = 0.0
bill_input_dropout = 0.0

init_leg_embedding_final = pd.DataFrame(np.random.uniform(-1.0, 1.0, size=(vote_data["J"], k_dim)))
init_leg_embedding_final.iloc[:, 0] = init_leg_embedding_final.iloc[:, 0].abs() * vote_data["init_embedding"]["init_value"]
max_norm = np.sqrt((init_leg_embedding_final ** 2).sum(axis=1)).max()
init_leg_embedding_final = init_leg_embedding_final / max_norm
if data_type == "test" and k_dim == 2:
    # init_leg_embedding_final.iloc[-1, 1] = -init_leg_embedding_final.iloc[:-1, ].product(axis=1).sum() / init_leg_embedding_final.iloc[-1, 0]
    # n = np.dot(init_leg_embedding_final.transpose(), init_leg_embedding_final)
    # m = np.eye(k_dim) * np.diag(n) - n
    # np.sqrt(np.sum(np.square(np.abs(m))))
    pass
    # init_leg_embedding_final.iloc[:, 0] = 0
    # init_leg_embedding_final.iloc[0, 0] = 0.5
    # init_leg_embedding_final.iloc[:, 1] = 0
    # init_leg_embedding_final.iloc[1, 1] = 0.5

if data_type == "cosponsor":
    # base_weight = 1
    # upweight_yes = 100
    # sample_weights = [base_weight, base_weight * upweight_yes]
    sample_weights = (1.0 * vote_data["y"].shape[0]) / (len(np.unique(vote_data["y"])) * np.bincount(vote_data["y"]))
if data_type == "votes":
    sample_weights = (1.0 * vote_data["y"].shape[0]) / (len(np.unique(vote_data["y"])) * np.bincount(vote_data["y"]))
if data_type == "test":
    sample_weights = [1, 1]

model = MOAmodels(n_leg, n_votes, k_dim=k_dim, init_leg_embedding=init_leg_embedding_final)

model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

callbacks = [EarlyStopping('loss', patience=10),
             GetBest(monitor='val_acc', verbose=1, mode='auto'),
             TerminateOnNaN()]
history = model.fit([vote_data["j"], vote_data["m"]] + vote_data["time_passed"], vote_data["y"], epochs=2000, batch_size=32768,
                    validation_split=0.2, verbose=2, callbacks=callbacks,
                    class_weight={0: sample_weights[0],
                                  1: sample_weights[1]})


# model.save(DATA_PATH + "keras_result.h5")
model.save_weights(DATA_PATH + 'my_model_weights.h5')

with open(DATA_PATH + "train_history.pkl", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
history_dict = history.history

# fitted_model = load_model(DATA_PATH + "keras_result.h5", custom_objects={"cust_reg": cust_reg})
model.load_weights(DATA_PATH + 'my_model_weights.h5')
fitted_model = model

with open(DATA_PATH + "train_history.pkl", 'rb') as file_pi:
        history_dict = pickle.load(file_pi)

# %matplotlib inline
# pd.DataFrame(fitted_model.layers[0].layers[0].get_weights()[0]).hist()
# pd.DataFrame(fitted_model.predict([vote_data["j"], vote_data["m"]] + vote_data["time_passed"]))[0].hist()

losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['loss']))],
                       'training': [loss for loss in history_dict['loss']],
                       'validation': [loss for loss in history_dict['val_loss']],
                       })
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("log loss")
ax.set_ylim([0.0, 3.0])


losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['acc']))],
                       'training': [loss for loss in history_dict['acc']],
                       'validation': [loss for loss in history_dict['val_acc']],
                       })
losses["validation"].max()
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0, 3.0])

if data_type == "votes" or data_type == "cosponsor":
    leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

if data_type == "test":
    test_actual = False
    if test_actual:
        leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0).reset_index()
    else:
        leg_data = pd.read_csv(DATA_PATH + f"/wnom{k_dim}D_results.csv", index_col=0)
        leg_data.index.name = "leg_id"
        leg_data = leg_data.reset_index()

    leg_data = leg_data.rename(columns={"index": "leg_id",
                                        "icpsrState": "state_icpsr",
                                        "coord1D.1": "nominate_dim1",
                                        "coord2D.1": "nominate_dim2",
                                        "coord3D.1": "nominate_dim3",
                                        "coord4D.1": "nominate_dim4",
                                        "coord5D.1": "nominate_dim5",
                                        "party.1": "party_code",
                                        "partyCode": "party_code"})
    leg_data["bioname"] = ""
    leg_data

    # fitted_model.get_layer("main_output").get_weights()[0]
    # fitted_model.get_layer("yes_term").get_weights()[0]
    # fitted_model.get_layer("no_term").get_weights()[0]
    # fitted_model.get_layer("yes_point").get_weights()[0]
    # np.exp(0.5)
col_names = (["leg_id", "state_icpsr", "bioname", "party_code"] +
             [f"nominate_dim{i}" for i in range(1, k_dim + 1)] +
             [f"coord{i}D" for i in range(1, k_dim + 1)])
leg_bio_data = leg_data[col_names].drop_duplicates()

fitted_model.get_layer("main_output").get_weights()[0]
fitted_model.layers[-2].get_weights()[0]


ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
drift_names = ["drift_{}".format(j) for j in range(1, k_time + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]

var_list = [f"nominate_dim{i}" for i in range(1, k_dim + 1)] + ideal_point_names + drift_names

cf_ideal_points = pd.DataFrame(fitted_model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])

leg_data_cf = pd.merge(leg_bio_data, cf_ideal_points, left_on="leg_id", right_index=True, how="inner")
if k_time > 0:
    cf_time_drift = pd.DataFrame(fitted_model.get_layer("ideal_points_time_{}".format(1)).get_weights()[0], columns=drift_names)
    cf_time_drift.index = pd.Series(cf_time_drift.index).map(vote_data["leg_crosswalk"])

    leg_data_cf = pd.merge(leg_data_cf, cf_time_drift, left_on="leg_id", right_index=True, how="inner")

leg_data_cf = pd.merge(leg_data_cf, first_session, left_on="leg_id", right_index=True, how="inner")

leg_data_cf.describe()
leg_data_cf.plot(kind="scatter", x="nominate_dim1", y="ideal_1")

full_sample = leg_data_cf[(leg_data_cf["first_session"] == 110) & (leg_data_cf["last_session"] == 115)]
full_sample.groupby("party_code").mean()
if k_time > 0:
    full_sample[full_sample["party_code"] == 100].sort_values("drift_1")
    leg_data_cf[leg_data_cf["party_code"] == 200].sort_values("drift_1")

leg_data_cf[var_list].corr()

yes_point = pd.DataFrame(fitted_model.get_layer("yes_point").get_weights()[0], columns=yes_point_names)
yes_point.index = pd.Series(yes_point.index).map(vote_data["vote_crosswalk"])
no_point = pd.DataFrame(fitted_model.get_layer("no_point").get_weights()[0], columns=no_point_names)
no_point.index = pd.Series(no_point.index).map(vote_data["vote_crosswalk"])

pol_pop = pd.merge(yes_point, no_point, left_index=True, right_index=True)
pol_pop.describe()
if data_type == "votes":
    pol_pop["temp_sess"] = pol_pop.index.str.slice(0, 3)
if data_type == "cosponsor":
    pol_pop["temp_sess"] = pol_pop.index.str[-3:]
if data_type == "test":
    pol_pop["temp_sess"] = 115
pol_pop.groupby(["temp_sess"]).mean()

fitted_model.layers
fitted_model.layers[-3].get_weights()

vote_df_eval = vote_df.copy()[["congress", "chamber", "leg_id", "first_session", "last_session", "time_passed"]].drop_duplicates()
vote_df_eval = pd.merge(vote_df_eval, cf_ideal_points, left_on="leg_id", right_index=True)
if k_time > 0:
    vote_df_eval = pd.merge(vote_df_eval, cf_time_drift, left_on="leg_id", right_index=True)

    # time_eval = [vote_df_eval[drift_names] * (vote_df_eval["time_passed"] ** j) for j in range(1, k_time + 1)]
    time_eval = vote_df_eval[drift_names].multiply(vote_df_eval["time_passed"], axis=0)
    time_final_names = ["time_ideal_{}".format(j) for j in range(1, k_dim + 1)]
    time_final = pd.DataFrame(vote_df_eval[ideal_point_names].values + time_eval.values,
                              columns=time_final_names,
                              index=time_eval.index)
    asdf = pd.merge(vote_df_eval, time_final, left_index=True, right_index=True)
    asdf.describe()
    zxcv = pd.merge(leg_bio_data, asdf, on="leg_id").sort_values("time_ideal_1")
    zxcv[zxcv["leg_id"] == 20735]
    zxcv.plot(kind="scatter", x="time_ideal_1", y="time_ideal_2")

if k_dim > 1:
    leg_data_cf.plot(kind="scatter", x="ideal_1", y="ideal_2")
leg_data_cf.plot(kind="scatter", x="nominate_dim1", y="ideal_1")

from pandas.plotting import scatter_matrix
# scatter_matrix(leg_data_cf[["nominate_dim1", "nominate_dim2", 0 , 1]],
#                alpha=0.2, figsize=(8, 8), diagonal='kde')

# leg_data_cf["color_list"] = "g"
# leg_data_cf.loc[leg_data_cf["party_code"] == 100, "color_list"] = "b"
# leg_data_cf.loc[leg_data_cf["party_code"] == 200, "color_list"] = "r"
# leg_data_cf.plot(kind="scatter", x="nominate_dim1", y=1, c="color_list")

# leg_data_cf.loc[leg_data_cf["party_code"] == 200, "ideal_1"].hist()
# leg_data_cf.loc[leg_data_cf["party_code"] == 200, "ideal_2"].hist()
# ax = leg_data_cf[leg_data_cf["party_code"] == 100].plot.scatter(x="ideal_1", y="ideal_2", color='DarkBlue', label='Dem')
# leg_data_cf[leg_data_cf["party_code"] == 200].plot.scatter(x="nominate_dim1", y="ideal_2", color='Red', label='Rep')

import seaborn as sns
sns.set(style="ticks")
sns.pairplot(leg_data_cf[leg_data_cf["party_code"].isin([100, 200])],
             hue="party_code", palette={100: "blue", 200: "red"},
             vars=var_list,
             diag_kind="kde", plot_kws=dict(alpha=0.25), markers=".")

leg_data_cf[var_list].corr()
leg_data_cf[[f"coord{i}D" for i in range(1, k_dim + 1)] + [f"nominate_dim{i}" for i in range(1, k_dim + 1)]].corr()
leg_data_cf[[f"coord{i}D" for i in range(1, k_dim + 1)] + [f"ideal_{i}" for i in range(1, k_dim + 1)]].corr()

leg_data_cf.groupby("party_code").mean()


leg_data_cf
A = leg_data_cf.dropna()
A["const"] = 1
x = np.linalg.lstsq(A[ideal_point_names].values, A[[f"nominate_dim{i}" for i in range(1, k_dim + 1)]].values)[0]

y_hat = A[ideal_point_names].values.dot(x)

pd.concat([A[[f"nominate_dim{i}" for i in range(1, k_dim + 1)]], pd.DataFrame(y_hat, index=A.index)], axis=1).corr()

midpoints = pd.DataFrame((fitted_model.get_layer("yes_point").get_weights()[0].dot(x) +
                          fitted_model.get_layer("no_point").get_weights()[0].dot(x)) / 2.0,
                          columns=["mid1", "mid2"],
                          index=vote_ids)

asdf = pd.merge(pd.read_csv(DATA_PATH + "/wnom2D_rollcalls.csv", index_col=0), midpoints, left_index=True, right_index=True)
asdf.plot(kind="scatter", x="midpoint1D", y="mid1")
asdf[["midpoint1D", "midpoint2D", "mid1", "mid2"]].corr()
sns.pairplot(asdf, vars=["midpoint1D", "midpoint2D", "mid1", "mid2"], diag_kind="kde", plot_kws=dict(alpha=0.25), markers=".")
