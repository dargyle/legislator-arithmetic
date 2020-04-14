import os
import numpy as np
import pandas as pd

import pickle

import seaborn as sns

from leg_math.keras_helpers import NNnominate
from leg_math.data_processing import process_data, drop_unanimous

from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.utils.vis_utils import model_to_dot

from leg_math.keras_helpers import GetBest, NNnominate
from leg_math.data_processing import process_data

from IPython.display import SVG

DATA_PATH = os.path.expanduser("~/data/leg_math/")
EU_PATH = os.path.expanduser("~/data/eu/")

most_recent_parties = pd.read_pickle(EU_PATH + "most_recent_parties.pkl")

eu_votes = pd.read_pickle(EU_PATH + "eu_votes.pkl")
vote_df = eu_votes.rename(columns={"voteid": "vote_id"})
vote_df = vote_df.drop(columns=["name", "obscure_id"])
vote_df = vote_df.dropna(subset=["mepid"]).copy()
vote_df["leg_id"] = vote_df["mepid"].astype(int).astype(str)
vote_df["vote"] = vote_df["vote_type"].map({"+": "yes", "-": "no", "0": np.nan})
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
               data_type="eu",
               congress_cutoff=0,
               k_dim=i,
               k_time=0,
               covariates_list=[],
               )
validation_split = 0.2
unanimity_check = True
k_dim = data_params["k_dim"]
k_time = data_params["k_time"]
covariates_list = data_params["covariates_list"]

# vote_data = process_data(**data_params, return_vote_df=False)
# vote_data, vote_df = process_data(**data_params, return_vote_df=True, unanimity_check=True)

leg_ids = vote_df["leg_id"].unique()
vote_ids = vote_df["vote_id"].unique()

if return_vote_df:
    vote_df_temp = vote_df.copy()
else:
    # Avoid unecessary data copy if not returning raw data
    vote_df_temp = vote_df
leg_crosswalk = pd.Series(leg_ids).to_dict()
leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
vote_crosswalk = pd.Series(vote_ids).to_dict()
vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

vote_df_temp["leg_id"] = vote_df_temp["leg_id"].map(leg_crosswalk_rev)
vote_df_temp["vote_id"] = vote_df_temp["vote_id"].map(vote_crosswalk_rev)
# Shuffle the order of the vote data
# THIS IS IMPORTANT, otherwise will_select just most recent bills
vote_df_temp = vote_df_temp.sample(frac=1, replace=False, random_state=42)

init_embedding = vote_df_temp[["leg_id", "init_value"]].drop_duplicates("leg_id").set_index("leg_id").sort_index()

if "vote_weight" not in vote_df_temp.columns:
    vote_df_temp["vote_weight"] = 1.0

assert not vote_df_temp.isnull().any().any(), "Missing value in data"

N = len(vote_df_temp)
key_index = round(validation_split * N)
print(f"key_index: {key_index}")

# Keep only votes that are valid in the dataset
train_data = vote_df_temp.iloc[:(N - key_index), :]
if unanimity_check:
    train_data = drop_unanimous(train_data, min_vote_count=10, unanimity_percentage=0.001)
# Ensure test data only contains valid entries
test_data = vote_df_temp.iloc[(N - key_index):, :]
test_data = test_data[test_data["leg_id"].isin(train_data["leg_id"])]
test_data = test_data[test_data["vote_id"].isin(train_data["vote_id"])]

time_passed_train = [(train_data["time_passed"] ** i).values for i in range(1, k_time + 1)]
time_passed_test = [(test_data["time_passed"] ** i).values for i in range(1, k_time + 1)]

vote_data = {'J': len(leg_ids),
             'M': len(vote_ids),
             'N': N,
             'j_train': train_data["leg_id"].values,
             'j_test': test_data["leg_id"].values,
             'm_train': train_data["vote_id"].values,
             'm_test': test_data["vote_id"].values,
             'y_train': train_data["vote"].astype(int).values,
             'y_test': test_data["vote"].astype(int).values,
             'time_passed_train': time_passed_train,
             'time_passed_test': time_passed_test,
             'init_embedding': init_embedding,
             'vote_crosswalk': vote_crosswalk,
             'leg_crosswalk': leg_crosswalk,
             'covariates_train': train_data[covariates_list].values,
             'covariates_test': test_data[covariates_list].values,
             'vote_weight_train': train_data["vote_weight"].values,
             'vote_weight_test': test_data["vote_weight"].values,
             }

# Export a pscl rollcall type object of the training data
# if data_type == 'test':
#     export_vote_df = vote_df_temp.iloc[:(N - key_index), :]
#     export_vote_df = export_vote_df[["leg_id", "vote_id", "vote"]]
#     export_vote_df["leg_id"] = export_vote_df["leg_id"].map(leg_crosswalk)
#     export_vote_df["vote_id"] = export_vote_df["vote_id"].map(vote_crosswalk)
#     roll_call = export_vote_df.set_index(["leg_id", "vote_id"])["vote"].map({1: 1, 0: 6}).unstack()
#     roll_call.fillna(9).astype(int).to_csv(DATA_PATH + "/test_votes.csv")

init_leg_embedding_final = pd.DataFrame(np.random.uniform(-1.0, 1.0, size=(vote_data["J"], k_dim)))
init_leg_embedding_final.iloc[:, 0] = init_leg_embedding_final.iloc[:, 0].abs() * vote_data["init_embedding"]["init_value"]
max_norm = np.sqrt((init_leg_embedding_final ** 2).sum(axis=1)).max()
init_leg_embedding_final = init_leg_embedding_final / (max_norm + 1e-7)

vote_data['init_embedding'] = init_leg_embedding_final


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

model.summary()

# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(vote_data["y_train"]))

callbacks = [EarlyStopping('val_loss', patience=7),
             GetBest(monitor='val_loss', verbose=1, mode='auto'),
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
valid_weight_count = ~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)
params_and_results["nonzero_dim_count"] = valid_weight_count.sum()

params_and_results = pd.Series(params_and_results)
print(params_and_results)

experimental_log_path = DATA_PATH + f'/experimental_log/{data_params["data_type"]}_model_metrics.csv'

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


losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['loss']))],
                       'training': [loss for loss in history_dict['loss']],
                       'validation': [loss for loss in history_dict['val_loss']],
                       })
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("log loss")
ax.set_ylim([0.0, 1.0])

losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['acc']))],
                       'training': [loss for loss in history_dict['acc']],
                       'validation': [loss for loss in history_dict['val_acc']],
                       })
losses["validation"].max()
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0, 1.0])


model.get_layer("main_output").get_weights()[0]
model.get_layer("wnom_term").get_weights()[0].round(5)
(~np.isclose(model.get_layer("wnom_term").get_weights()[0], 0)).sum()

ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
drift_names = ["drift_{}".format(j) for j in range(1, k_time + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]

if data_params["data_type"] == "test":
    var_list = [f"nominate_dim{i}" for i in range(1, k_dim + 1)] + ideal_point_names + drift_names
if data_params["data_type"] == "votes" or data_params["data_type"] == "cosponsor" or data_params["data_type"] == "eu":
    var_list = [f"nominate_dim{i}" for i in range(1, 3)] + ideal_point_names + drift_names

cf_ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
cf_ideal_points["ideal_2"] = cf_ideal_points["ideal_2"] * -1
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])
cf_ideal_points.to_pickle(EU_PATH + 'eu_ideal_points.pkl')
cf_ideal_points.to_csv(EU_PATH + 'eu_ideal_points.csv')

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
g.figure.savefig(EU_PATH + "eu_ideologies.pdf")
