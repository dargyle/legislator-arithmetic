import os
import numpy as np
import pandas as pd

import pickle

from keras.layers import Embedding, Reshape, Merge, Dropout, SpatialDropout1D, Dense, Flatten, Input, Dot, LSTM, Add, Conv1D, MaxPooling1D, Concatenate,  Multiply
from keras.models import Sequential, Model
from keras import regularizers

from keras.models import load_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

DATA_PATH = os.path.expanduser("~/data/leg_math/")

vote_df_temp = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")

leg_ids = vote_df_temp["leg_id"].unique()
vote_ids = vote_df_temp["vote_id"].unique()

leg_crosswalk = pd.Series(leg_ids).to_dict()
leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
vote_crosswalk = pd.Series(vote_ids).to_dict()
vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

vote_df_temp["leg_id"] = vote_df_temp["leg_id"].map(leg_crosswalk_rev)
vote_df_temp["vote_id"] = vote_df_temp["vote_id"].map(vote_crosswalk_rev)
# Shuffle the order of the vote data
vote_df_temp = vote_df_temp.sample(frac=1, replace=False)

init_embedding = vote_df_temp[["leg_id", "init_value"]].drop_duplicates().set_index("leg_id").sort_index()

vote_data = {'J': len(leg_ids),
             'M': len(vote_ids),
             'N': len(vote_df_temp),
             'j': vote_df_temp["leg_id"].values,
             'm': vote_df_temp["vote_id"].values,
             'y': vote_df_temp["vote"].astype(int).values,
             'init_embedding': init_embedding,
             'vote_crosswalk': vote_crosswalk,
             'leg_crosswalk': leg_crosswalk}

print("N Legislators: {}".format(vote_data["J"]))
print("N Votes: {}".format(vote_data["M"]))
print("N: {}".format(vote_data["N"]))

print("Let's do a keras cf example")
print(vote_data["y"].mean())

n_users = vote_data["J"]
m_items = vote_data["M"]
k_dim = 5

init_embedding_final = pd.concat([vote_data["init_embedding"]["init_value"] * np.random.rand(vote_data["J"]) for j in range(k_dim)], axis=1)

# Switch to the functional api (current version)
leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
P = Embedding(input_dim=n_users, output_dim=k_dim, input_length=1, weights=[init_embedding_final.values])(leg_input)
flat_P = Reshape((k_dim,))(P)
bill_input = Input(shape=(1, ), dtype="int32", name="bill_input")
Q = Embedding(input_dim=m_items, output_dim=k_dim, input_length=1)(bill_input)
flat_Q = Reshape((k_dim,))(Q)
combined = Dot(axes=1)([flat_P, flat_Q])
main_output = Dense(1, activation="sigmoid", name="main_output")(combined)

model = Model(inputs=[leg_input, bill_input], outputs=[main_output])

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

sample_weights = (1.0 * vote_data["y"].shape[0]) / (len(np.unique(vote_data["y"])) * np.bincount(vote_data["y"]))

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
# MODEL_WEIGHTS_FILE = ("~/temp/" + "collab_filter_example.h5")
# callbacks = [EarlyStopping('val_loss', patience=5),
#              ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
# history = model.fit([vote_data["j"], vote_data["m"]], vote_data["y"], epochs=100,
#                     validation_split=.2, verbose=2, callbacks=callbacks,
#                     class_weight={0: sample_weights[0],
#                                   1: sample_weights[1]})

callbacks = [EarlyStopping('val_loss', patience=25)]
history = model.fit([vote_data["j"], vote_data["m"]], vote_data["y"], epochs=200, batch_size=8192,
                    validation_split=.2, verbose=2, callbacks=callbacks,
                    class_weight={0: sample_weights[0],
                                  1: sample_weights[1]})

model.save(DATA_PATH + "keras_result.h5")

with open(DATA_PATH + "train_history.pkl", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

fitted_model = load_model(DATA_PATH + "keras_result.h5")

with open(DATA_PATH + "train_history.pkl", 'rb') as file_pi:
        history_dict = pickle.load(file_pi)

# %matplotlib inline
# pd.DataFrame(fitted_model.layers[0].layers[0].get_weights()[0]).hist()
pd.DataFrame(fitted_model.predict([vote_data["j"], vote_data["m"]]))[0].hist()

losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['loss']))],
                       'training': [loss for loss in history_dict['loss']],
                       'validation': [loss for loss in history_dict['val_loss']]})
ax = losses.plot(x='epoch', figsize={7, 10}, grid=True)
ax.set_ylabel("log loss")
ax.set_ylim([0.0, 3.0])


losses = pd.DataFrame({'epoch': [i + 1 for i in range(len(history_dict['acc']))],
                       'training': [loss for loss in history_dict['acc']],
                       'validation': [loss for loss in history_dict['val_acc']]})
ax = losses.plot(x='epoch', figsize={7, 10}, grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0, 3.0])

leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

pd.DataFrame(fitted_model.layers[2].get_weights()[0])
cf_ideal_points = pd.DataFrame(fitted_model.layers[2].get_weights()[0])
cf_ideal_points.index = pd.Series(cf_ideal_points.index).map(vote_data["leg_crosswalk"])
leg_data_cf = pd.merge(leg_data, cf_ideal_points, left_on="leg_id", right_index=True)
leg_data_cf.plot(kind="scatter", x="nominate_dim1", y=0)

from pandas.plotting import scatter_matrix
scatter_matrix(leg_data_cf[["nominate_dim1", "nominate_dim2", 0 , 1]],
               alpha=0.2, figsize=(8, 8), diagonal='kde')

# leg_data_cf["color_list"] = "g"
# leg_data_cf.loc[leg_data_cf["party_code"] == 100, "color_list"] = "b"
# leg_data_cf.loc[leg_data_cf["party_code"] == 200, "color_list"] = "r"
# leg_data_cf.plot(kind="scatter", x="nominate_dim1", y=1, c="color_list")

leg_data_cf.loc[leg_data_cf["party_code"] == 200, 0].hist()
leg_data_cf.loc[leg_data_cf["party_code"] == 200, 1].hist()
ax = leg_data_cf[leg_data_cf["party_code"] == 100].plot.scatter(x=0, y=1, color='DarkBlue', label='Dem', xlim=(-4,4), ylim=(-4,4))
leg_data_cf[leg_data_cf["party_code"] == 200].plot.scatter(x="nominate_dim1", y=1, color='Red', label='Rep')

import seaborn as sns
sns.set(style="ticks")

sns.pairplot(leg_data_cf[leg_data_cf["party_code"].isin([100, 200])], hue="party_code", vars=["nominate_dim1", "nominate_dim2", 0 , 1], diag_kind="kde", plot_kws=dict(alpha=0.25), markers=".")

leg_data_cf.groupby("party_code").mean()

leg_data_cf.plot(kind="scatter", x=0, y=1)
leg_data_cf[0].hist()
