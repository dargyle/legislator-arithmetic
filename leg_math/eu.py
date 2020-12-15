import os
import numpy as np
import pandas as pd

import pickle

import seaborn as sns

from leg_math.keras_helpers import NNnominate
from data_generation.data_processing import process_data, drop_unanimous

from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.utils import model_to_dot

from leg_math.keras_helpers import GetBest, NNnominate

from IPython.display import SVG

from constants import DATA_PATH

EU_PATH = DATA_PATH + '/eu/'
if not os.path.exists(EU_PATH):
    os.makedirs(EU_PATH)

most_recent_parties = pd.read_pickle(EU_PATH + "most_recent_parties.pkl")

eu_votes = pd.read_pickle(EU_PATH + "eu_votes.pkl")
eu_votes["voteid"] = eu_votes["voteid"].astype(str)

vote_metadata = pd.read_pickle(EU_PATH + "eu_vote_metadata.pkl")
vote_metadata["voteid"] = vote_metadata["voteid"].astype(str)
# vote_metadata['time_passed'] = vote_metadata['ts'].dt.year - 2004
vote_metadata["congress"] = vote_metadata['ts'].dt.year
vote_time_passed = vote_metadata[["voteid", "congress"]]
eu_votes = pd.merge(eu_votes, vote_time_passed, on='voteid')

vote_df = eu_votes.rename(columns={"voteid": "vote_id"})
vote_df = vote_df.drop(columns=["name", "obscure_id"])
vote_df = vote_df.dropna(subset=["mepid"]).copy()
vote_df["leg_id"] = vote_df["mepid"].astype(int).astype(str)
vote_df["vote"] = vote_df["vote_type"].map({"+": 1, "-": 0, "0": np.nan})
vote_df = vote_df.dropna().copy()
vote_df["vote"] = vote_df["vote"].astype(int)

vote_df = pd.merge(vote_df,
                   most_recent_parties[["person_id", "init_value"]],
                   left_on='leg_id',
                   right_on="person_id",
                   how='left')

i = 2
return_vote_df = True

data_params = dict(
               vote_df=vote_df,
               congress_cutoff=0,
               k_dim=i,
               k_time=1,
               covariates_list=[],
               )
validation_split = 0.2
unanimity_check = True
k_dim = data_params["k_dim"]
k_time = data_params["k_time"]
covariates_list = data_params["covariates_list"]

vote_data = process_data(**data_params, return_vote_df=False)

model_params = {
                "n_leg": vote_data["J"],
                "n_votes": vote_data["M"],
                "k_dim": data_params["k_dim"],
                "k_time": data_params["k_time"],
                "init_leg_embedding": vote_data["init_embedding"],
                "yes_point_dropout": 0.0,
                "no_point_dropout": 0.0,
                "combined_dropout": 0.0,
                "dropout_type": "timestep",
                "gaussian_noise": 0.05,
                "covariates_list": data_params["covariates_list"],
                }

model = NNnominate(**model_params)

model.summary()

# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

weight_by_frequency = False
if weight_by_frequency:
    sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(np.squeeze(vote_data["y_train"])))
else:
    sample_weights = {k: 1 for k in np.unique(vote_data["y_train"])}

callbacks = [EarlyStopping('val_loss', patience=7, restore_best_weights=True),
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
                    validation_data=(x_test, vote_data["y_test"]), verbose=2, callbacks=callbacks,
                    class_weight={0: sample_weights[0],
                                  1: sample_weights[1]})

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
model.get_layer("wnom_term").get_weights()[0]
np.log(model.get_layer("wnom_term").get_weights()[0])
valid_weight_count = ~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)
params_and_results["nonzero_dim_count"] = valid_weight_count.sum()

params_and_results = pd.Series(params_and_results)
print(params_and_results)

# experimental_log_path = DATA_PATH + f'/experimental_log/{data_params["data_type"]}_model_metrics.csv'
#
# try:
#     exp_log = pd.read_csv(experimental_log_path, index_col=False)
#     exp_log = exp_log.append(params_and_results.to_frame().transpose())
# except IOError:
#     exp_log = params_and_results.to_frame().transpose()
#
#
# exp_log.to_csv(experimental_log_path, index=False)
#
# # model.save(DATA_PATH + '/models/{data_type}_model_{congress_cutoff}_{k_dim}_{k_time}.h5'.format(**data_params))
# fname_weights = '/models/{data_type}_model_weights_{congress_cutoff}_{k_dim}_{k_time}.h5'.format(**data_params)
# model.save_weights(DATA_PATH + fname_weights)
#
# fname_history = '/models/{data_type}_train_history_{congress_cutoff}_{k_dim}_{k_time}.pkl'.format(**data_params)
# with open(DATA_PATH + fname_history, 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
history_dict = history.history


losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['loss']))],
                       'training': [loss for loss in history_dict['loss']],
                       'validation': [loss for loss in history_dict['val_loss']],
                       })
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("log loss")
ax.set_ylim([0.0, 1.0])

losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['accuracy']))],
                       'training': [loss for loss in history_dict['accuracy']],
                       'validation': [loss for loss in history_dict['val_accuracy']],
                       })
losses["validation"].max()
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0, 1.0])


model.get_layer("main_output").get_weights()[0]
model.get_layer("wnom_term").get_weights()[0].round(5)
(~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)).sum()

ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
# TODO: Fix to generalize to k_time > 1
drift_names = ["drift_{}".format(j) for j in range(1, k_dim + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]

# if data_params["data_type"] == "test":
#     var_list = [f"nominate_dim{i}" for i in range(1, k_dim + 1)] + ideal_point_names + drift_names
# if data_params["data_type"] == "votes" or data_params["data_type"] == "cosponsor" or data_params["data_type"] == "eu":
var_list = [f"nominate_dim{i}" for i in range(1, 3)] + ideal_point_names + drift_names

cf_ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
cf_ideal_points["ideal_2"] = cf_ideal_points["ideal_2"] * -1
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])
# cf_ideal_points.to_pickle(EU_PATH + 'eu_ideal_points.pkl')
# cf_ideal_points.to_csv(EU_PATH + 'eu_ideal_points.csv')

cf_ideal_points["ideal_1"].hist()
cf_ideal_points["ideal_2"].hist()
cf_ideal_points.plot(kind="scatter", x='ideal_1', y='ideal_2')

# Flip the axis to align with our common perception

# eu_parties = pd.read_pickle(EU_PATH + "most_recent_parties.pkl")
# eu_parties["active"] = (eu_parties["end"].dt.date > pd.datetime.now().date())
# eu_parties["active"] = eu_parties["active"].map({False: "inactive", True: "active"})
# eu_parties["person_id"] = eu_parties["person_id"].astype(str)

leg_data = pd.merge(most_recent_parties, cf_ideal_points, left_on="person_id", right_index=True)
leg_data["person_id"].duplicated().any()
# membership_counts = leg_data["party"].value_counts()
#

leg_data.plot(kind="scatter", x='ideal_1', y='ideal_2')

sns.set(rc={'figure.figsize': (8.0, 8.0)})

hue_map = {'PPE': '#3399FF',
           'S&D': '#FF0000',
           'RE': 'gold',
           'Verts/ALE': '#009900',
           'ID': '#2B3856',
           'CRE': '#0054A5',
           'GUE/NGL': '#990000',
           'NA': '#999999',
           'EFDD': '#24B9B9',
           }


g = sns.scatterplot(x="ideal_1", y="ideal_2", hue="party_plot", data=leg_data, palette=hue_map, style="active", style_order=["active", "inactive"], alpha=0.75)
g.figure.savefig(EU_PATH + "eu_ideologies_nn_nominate.png")


from leg_math.keras_helpers import NNitemresponse
model_params.pop("yes_point_dropout")
model_params.pop("no_point_dropout")
model_params["batch_normalize"] = True
# model_params["k_dim"] = 1
# model_params["k_time"] = 1
item_model = NNitemresponse(**model_params)

item_model.summary()

# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
item_model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
item_history = item_model.fit(x_train, vote_data["y_train"], epochs=5000, batch_size=32768,
                              validation_data=(x_test, vote_data["y_test"]), verbose=2, callbacks=callbacks,
                              class_weight={0: sample_weights[0],
                                            1: sample_weights[1]})
train_metrics = item_model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
test_metrics = item_model.evaluate(x_test, vote_data["y_test"], batch_size=10000)

item_model.summary()
drift_weights = pd.DataFrame(item_model.get_layer("ideal_points_time_1").get_weights()[0], columns=drift_names)
drift_weights.index = pd.Series(drift_weights.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.DataFrame(item_model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
# cf_ideal_points["ideal_2"] = cf_ideal_points["ideal_2"] * -1
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.merge(cf_ideal_points, drift_weights, left_index=True, right_index=True)

leg_data = pd.merge(most_recent_parties, cf_ideal_points, left_on="person_id", right_index=True)
leg_data["person_id"].duplicated().any()
# membership_counts = leg_data["party"].value_counts()
#
leg_data.plot(kind="scatter", x='ideal_1', y='ideal_2')

sns.set(rc={'figure.figsize': (8.0, 8.0)})

hue_map = {'PPE': '#3399FF',
           'S&D': '#FF0000',
           'RE': 'gold',
           'Verts/ALE': '#009900',
           'ID': '#2B3856',
           'CRE': '#0054A5',
           'GUE/NGL': '#990000',
           'NA': '#999999',
           'EFDD': '#24B9B9',
           }

cf_ideal_points.hist()
g = sns.scatterplot(x="ideal_1", y="ideal_2", hue="party_plot", data=leg_data, palette=hue_map, style="active", style_order=["active", "inactive"], alpha=0.75)
g.figure.savefig(EU_PATH + "eu_ideologies_item_response.png")

item_model.get_layer("main_output").get_weights()[0]
pd.DataFrame(item_model.get_layer("polarity").get_weights()[0]).plot(kind="scatter", x=0, y=1, alpha=0.25)
pd.DataFrame(item_model.get_layer("popularity").get_weights()[0]).hist()

eu_leg_data = pd.read_pickle(EU_PATH + "leg_data.pkl")[["person_id", "name_full", "home_country", "home_party"]]
eu_leg_data = pd.merge(eu_leg_data, leg_data[["person_id", "active", "party_plot"] + ideal_point_names + drift_names], on="person_id")

dynamic_leg_data = pd.merge(eu_leg_data, vote_df[["leg_id", "congress"]].drop_duplicates(), left_on="person_id", right_on="leg_id")

first_session = dynamic_leg_data.groupby("leg_id")[["congress"]].agg(["min", "max"])
first_session.columns = ["first_session", "last_session"]

dynamic_leg_data = pd.merge(dynamic_leg_data, first_session, left_on="leg_id", right_index=True)
dynamic_leg_data["time_passed"] = dynamic_leg_data["congress"] - dynamic_leg_data["first_session"]

dynamic_leg_data["ideal_1_time"] = dynamic_leg_data["ideal_1"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_1"]
dynamic_leg_data["ideal_1_time"].hist()
dynamic_leg_data["ideal_2_time"] = dynamic_leg_data["ideal_2"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_2"]

dynamic_leg_data.to_pickle(EU_PATH + "dynamic_leg_data.pkl")

import plotly.graph_objects as go
leg_ids = dynamic_leg_data.loc[dynamic_leg_data["active"] == "active", "leg_id"].unique()
leg_id = leg_ids[3]
dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id]

fig = go.Figure()
for leg_id in leg_ids:
    fig.add_trace(go.Scatter(x=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_1_time"],
                             y=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_2_time"],
                             mode='lines+markers',
                             line=dict(color=hue_map[dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "party_plot"].iloc[0]])))
fig.show()

vote_df.groupby(["vote_id", "party_abbrev", "vote"])["mepid"].count().unstack(["party_abbrev", "vote"])







drift_weights = pd.DataFrame(model.get_layer("ideal_points_time_1").get_weights()[0], columns=drift_names)
drift_weights.index = pd.Series(drift_weights.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
cf_ideal_points["ideal_2"] = cf_ideal_points["ideal_2"] * -1
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.merge(cf_ideal_points, drift_weights, left_index=True, right_index=True)

leg_data = pd.merge(most_recent_parties, cf_ideal_points, left_on="person_id", right_index=True)
leg_data["person_id"].duplicated().any()
# membership_counts = leg_data["party"].value_counts()
#
leg_data.plot(kind="scatter", x='ideal_1', y='ideal_2')

sns.set(rc={'figure.figsize': (8.0, 8.0)})

hue_map = {'PPE': '#3399FF',
           'S&D': '#FF0000',
           'RE': 'gold',
           'Verts/ALE': '#009900',
           'ID': '#2B3856',
           'CRE': '#0054A5',
           'GUE/NGL': '#990000',
           'NA': '#999999',
           'EFDD': '#24B9B9',
           }

cf_ideal_points.hist()
g = sns.scatterplot(x="ideal_1", y="ideal_2", hue="party_plot", data=leg_data, palette=hue_map, style="active", style_order=["active", "inactive"], alpha=0.75)
g.figure.savefig(EU_PATH + "eu_ideologies_item_response.png")

eu_leg_data = pd.read_pickle(EU_PATH + "leg_data.pkl")[["person_id", "name_full", "home_country", "home_party"]]
eu_leg_data = pd.merge(eu_leg_data, leg_data[["person_id", "active", "party_plot"] + ideal_point_names + drift_names], on="person_id")

dynamic_leg_data = pd.merge(eu_leg_data, vote_df[["leg_id", "congress"]].drop_duplicates(), left_on="person_id", right_on="leg_id")

first_session = dynamic_leg_data.groupby("leg_id")[["congress"]].agg(["min", "max"])
first_session.columns = ["first_session", "last_session"]

dynamic_leg_data = pd.merge(dynamic_leg_data, first_session, left_on="leg_id", right_index=True)
dynamic_leg_data["time_passed"] = dynamic_leg_data["congress"] - dynamic_leg_data["first_session"]

dynamic_leg_data["ideal_1_time"] = dynamic_leg_data["ideal_1"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_1"]
dynamic_leg_data["ideal_2_time"] = dynamic_leg_data["ideal_2"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_2"]

dynamic_leg_data.to_pickle(EU_PATH + "dynamic_leg_data.pkl")

import plotly.graph_objects as go
leg_ids = dynamic_leg_data.loc[dynamic_leg_data["active"] == "active", "leg_id"].unique()
leg_id = leg_ids[3]
dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id]

fig = go.Figure()
for leg_id in leg_ids:
    fig.add_trace(go.Scatter(x=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_1_time"],
                             y=dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "ideal_2_time"],
                             mode='lines+markers',
                             line=dict(color=hue_map[dynamic_leg_data.loc[dynamic_leg_data["leg_id"] == leg_id, "party_plot"].iloc[0]])))
fig.show()
