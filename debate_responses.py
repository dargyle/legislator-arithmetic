import os
import numpy as np
import pandas as pd

import pickle

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint

from data_generation.data_processing import process_data, format_model_data
from data_generation.random_votes import generate_nominate_votes

from leg_math.keras_helpers import GetBest, NNitemresponse

from scipy import stats

from sklearn.metrics import log_loss, accuracy_score

from constants import DATA_PATH

DEBATE_PATH = os.path.expanduser('~/research/debate-ideal-points/data/')

vote_df = pd.read_pickle(DEBATE_PATH + "vote_df_test.pkl")
init_mean = pd.read_pickle(DEBATE_PATH + "init_mean_test.pkl")
init_mean.name = "init_value"
vote_counts = pd.read_pickle(DEBATE_PATH + "vote_counts_test.pkl")

vote_df["congress"] = 0
vote_df["vote"] = vote_df["vote"] - 1
vote_df = pd.merge(vote_df, init_mean, left_on='leg_id', right_index=True)

print("Get nn_estimate")
# Process the vote_df into the model data format
i = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=0,
               k_dim=i,
               k_time=0,
               covariates_list=[],
               unanimity_check=False,
               validation_split=0.05,
               )
vote_data = process_data(**data_params)

y_train = vote_data["y_train"]
vote_data["y_train"] = [1 * (vote_data["y_train"] >= 1), 1 * (vote_data["y_train"] >= 2)]
vote_data["y_test"] = [1 * (vote_data["y_test"] >= 1), 1 * (vote_data["y_test"] >= 2)]
vote_data["init_embedding"]
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
                "k_out": 2,
                }


model = NNitemresponse(**model_params)

model.summary()
SVG(model_to_dot(model, dpi=60, show_shapes=True).create(prog='dot', format='svg'))

# opt = tfp.optimizer.VariationalSGD(batch_size=1024,
#                                    total_num_examples=vote_data["N"],
#                                    # use_single_learning_rate=True,
#                                    burnin=100,
#                                    max_learning_rate=3.0,
#                                    burnin_max_learning_rate=3.0,
#                                    preconditioner_decay_rate=0.95,
#                                    )
opt = tf.keras.optimizers.Nadam()
model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=opt)

callbacks = [
             EarlyStopping('val_loss', patience=20, restore_best_weights=True),
             # EarlyStopping('val_loss', patience=250, restore_best_weights=True),
             # GetBest(monitor='val_loss', verbose=1, mode='auto'),
             # ModelCheckpoint(DATA_PATH + '/temp/model_weights_{epoch}.hdf5'),
             TerminateOnNaN(),
             ]
history = model.fit(x_train, vote_data["y_train"], validation_data=(x_test, vote_data["y_test"]), epochs=5000, batch_size=1024, verbose=2, callbacks=callbacks)
model.predict(x_test)

loss = tf.keras.losses.CategoricalCrossentropy()
loss = tf.keras.losses.BinaryCrossentropy()
loss(vote_data["y_test"], model.predict(x_test))

acc = tf.metrics.Accuracy()

pred = pd.DataFrame(np.concatenate(model.predict(x_test), axis=1))
loss(vote_data["y_test"], pred).numpy()
pred.sum(axis=1)
acc(vote_data["y_test"], 1 * (pred > 0.5))

pd.DataFrame(np.concatenate(vote_data["y_test"], axis=1)).describe()
pd.DataFrame(pred).sort_values(0)
pd.merge(pd.DataFrame(model.predict(x_test)), pd.DataFrame(vote_data["y_test"]), left_index=True, right_index=True).sort_values(["0_y", "1_y"])
pd.DataFrame(pred).describe()
pd.DataFrame(model.get_layer("ideal_points").get_weights()[0]).hist()

train_metrics = model.evaluate(x_train, vote_data["y_train"], batch_size=10000)
acc(vote_data["y_train"], 1 * (model.predict(x_train) > 0.5))
test_metrics = model.evaluate(x_test, vote_data["y_test"], batch_size=10000)
acc(vote_data["y_test"], 1 * (model.predict(x_test) > 0.5))

pred
result = pd.DataFrame(pred)
result["prediction"] = np.nan
result.loc[(result[0] < 0.5) & result["prediction"].isnull(), "prediction"] = 1
result.loc[(result[1] < 0.5) & result["prediction"].isnull(), "prediction"] = 2
# result.loc[((result[0] >= 0.5) & (result[1] < 0.5)), "prediction"] = 1
# result.loc[(result[1] >= 0.5) & (result[0] < 0.5), "prediction"] = 2
result["prediction"].fillna(0).value_counts()
result.describe()
result[result[1] > 0.5]

result.sum(axis=1)
result.sort_values(by=1)

asdf = pd.DataFrame(vote_data["y_train"])
asdf["temp"] = 1
asdf.groupby([0, 1]).sum()

result.describe()
