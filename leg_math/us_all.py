import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint

from data_generation.data_processing import process_data, format_model_data
from data_generation.random_votes import generate_nominate_votes

from leg_math.keras_helpers import GetBest, NNitemresponse, NNnominate

from scipy import stats

from sklearn.metrics import log_loss, accuracy_score

from constants import DATA_PATH


vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

print("Get nn_estimate")
# Process the vote_df into the model data format
i = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=0,
               k_dim=i,
               k_time=1,
               covariates_list=[],
               )
vote_data = process_data(**data_params)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

model_params = {
                "n_leg": vote_data["J"],
                "n_votes": vote_data["M"],
                "k_dim": data_params["k_dim"],
                "k_time": data_params["k_time"],
                "init_leg_embedding": vote_data["init_embedding"],
                # "yes_point_dropout": 0.0,
                # "no_point_dropout": 0.0,
                "combined_dropout": 0.0,
                "dropout_type": "timestep",
                "gaussian_noise": 0.0,
                "covariates_list": data_params["covariates_list"],
                # "main_activation": "gaussian",
                }

# model = NNitemresponse(**model_params)
model = NNnominate(**model_params)

model.summary()
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# opt = tfp.optimizer.VariationalSGD(batch_size=1024,
#                                    total_num_examples=vote_data["N"],
#                                    # use_single_learning_rate=True,
#                                    burnin=100,
#                                    max_learning_rate=3.0,
#                                    burnin_max_learning_rate=3.0,
#                                    preconditioner_decay_rate=0.95,
#                                    )
# opt = tf.keras.optimizers.Nadam()
opt = tfp.optimizer.VariationalSGD(batch_size=1, total_num_examples=vote_data["N"])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

callbacks = [
             EarlyStopping('val_loss', patience=20, restore_best_weights=True),
             # EarlyStopping('val_loss', patience=250, restore_best_weights=True),
             # GetBest(monitor='val_loss', verbose=1, mode='auto'),
             ModelCheckpoint(DATA_PATH + '/temp/model_weights_{epoch}.hdf5'),
             TerminateOnNaN()]
history = model.fit(x_train, vote_data["y_train"], epochs=5000, batch_size=1,
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

k_dim = 2
ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
# TODO: Fix to generalize to k_time > 1
drift_names = ["drift_{}".format(j) for j in range(1, k_dim + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]

# if data_params["data_type"] == "test":
#     var_list = [f"nominate_dim{i}" for i in range(1, k_dim + 1)] + ideal_point_names + drift_names
# if data_params["data_type"] == "votes" or data_params["data_type"] == "cosponsor" or data_params["data_type"] == "eu":
var_list = [f"nominate_dim{i}" for i in range(1, 3)] + ideal_point_names + drift_names

drift_weights = pd.DataFrame(model.get_layer("ideal_points_time_1").get_weights()[0], columns=drift_names)
drift_weights.index = pd.Series(drift_weights.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.DataFrame(model.get_layer("ideal_points").get_weights()[0], columns=ideal_point_names)
# cf_ideal_points["ideal_2"] = cf_ideal_points["ideal_2"] * -1
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])

cf_ideal_points = pd.merge(cf_ideal_points, drift_weights, left_index=True, right_index=True)

leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")
party_data = pd.read_feather(DATA_PATH + "party_data.feather")

leg_data = pd.merge(leg_data, party_data[["party_code", "party_name"]].drop_duplicates(), on="party_code")
leg_data = leg_data[leg_data["chamber"] == "Senate"]
leg_data["leg_id"].duplicated().any()
# membership_counts = leg_data["party"].value_counts()

leg_data = pd.merge(leg_data, cf_ideal_points, left_on="leg_id", right_index=True)
leg_data.plot(kind="scatter", x='ideal_1', y='ideal_2')

first_session = leg_data.groupby("leg_id")[["congress"]].agg(["min", "max"])
first_session.columns = ["first_session", "last_session"]

dynamic_leg_data = pd.merge(leg_data, first_session, left_on="leg_id", right_index=True)
dynamic_leg_data["time_passed"] = dynamic_leg_data["congress"] - dynamic_leg_data["first_session"]

dynamic_leg_data["ideal_1_time"] = dynamic_leg_data["ideal_1"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_1"]
dynamic_leg_data["ideal_1_time"].hist()
dynamic_leg_data["ideal_2_time"] = dynamic_leg_data["ideal_2"] + dynamic_leg_data["time_passed"] * dynamic_leg_data["drift_2"]

import plotly_express as px
dynamic_leg_data["party_plot"] = "Other"
dynamic_leg_data.loc[dynamic_leg_data["party_name"] == "Democrat", "party_plot"] = "Democrat"
dynamic_leg_data.loc[dynamic_leg_data["party_name"] == "Republican", "party_plot"] = "Republican"

hue_map = {'Democrat': 'blue',
           'Republican': 'red',
           'Other': 'grey',
           }

fig = px.line(dynamic_leg_data[dynamic_leg_data["last_session"] >= 116], x="ideal_1_time", y="ideal_2_time",
              color="party_plot", line_group="leg_id", hover_name='bioname',
              color_discrete_map=hue_map,
              hover_data={'state_abbrev': True, "congress": True, "party_plot": False, "party_name": True}, width=1000, height=800)
fig.update_traces(mode='lines+markers')
fig.show()
fig.write_html(DATA_PATH + "us_dynamic_viz_current.html")

fig = px.line(dynamic_leg_data, x="ideal_1_time", y="ideal_2_time",
              color="party_plot", line_group="leg_id", hover_name='bioname',
              color_discrete_map=hue_map,
              hover_data={'state_abbrev': True, "congress": True, "party_plot": False, "party_name": True}, width=1000, height=800)
fig.update_traces(mode='lines+markers')
fig.show()
fig.write_html(DATA_PATH + "us_dynamic_viz_all.html")
