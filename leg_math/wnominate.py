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

from keras.models import load_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras import backend as K

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
# # run -f has_inf_or_nan
# K.set_session(sess)


class GetBest(Callback):
    """Get the best model at the end of training.
    https://github.com/keras-team/keras/issues/2768

    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)


class OrthReg(Regularizer):
    """Orthogonality regularizers

    # Arguments
        rf: Float; rf regularization factor.
    """
    # https://stackoverflow.com/questions/42911671/how-can-i-add-orthogonality-regularization-in-keras
    # m = K.dot(K.transpose(w), w) - K.eye(w[1].shape[0])

    def __init__(self, rf=1.0):
        self.rf = K.cast_to_floatx(rf)

    def __call__(self, x):
        regularization = 0.0
        m = K.dot(K.transpose(x), x)
        n = m - K.eye(K.int_shape(m)[0])
        norm = K.sqrt(self.rf * K.sum(K.square(K.abs(n))))
        regularization += norm
        print(regularization)
        return regularization

    def get_config(self):
        return {'rf': float(self.rf)}


# class UnitSphere(Regularizer):
#     """Orthogonality regularizers
#
#     # Arguments
#         rf: Float; rf regularization factor.
#     """
#     # https://stackoverflow.com/questions/42911671/how-can-i-add-orthogonality-regularization-in-keras
#     # m = K.dot(K.transpose(w), w) - K.eye(w[1].shape[0])
#
#     def __init__(self, rf=1.0):
#         self.rf = K.cast_to_floatx(rf)
#
#     def __call__(self, x):
#         regularization = 0.0
#         # Sum the square of the individual's ideal points, then subtract 1
#         temp_sum = K.sum(K.square(x), axis=0)
#         # relu sets negative to zero
#         # https://stackoverflow.com/questions/41043894/setting-all-negative-values-of-a-tensor-to-zero-in-tensorflow
#         # reg_term = K.sqrt(self.rf * K.sum(K.relu(temp_sum)))
#         reg_term = K.cast(temp_sum > 0.0, temp_sum.dtype) * temp_sum
#         reg_term_sum = K.sqrt(self.rf * K.sum(reg_term))
#         # if np.isfinite(reg_term):
#         regularization += reg_term_sum
#         print(regularization)
#         return regularization
#
#     def get_config(self):
#         return {'rf': float(self.rf)}


class UnitSphere(Constraint):
    """Orthogonality constraint

    # Arguments
        rf: Float; rf regularization factor.
    """
    def __call__(self, w):
        temp_sum = K.sum(K.square(w), axis=0)
        w *= K.cast(K.less_equal(temp_sum, 1.0), K.floatx())
        return w

# asdf = fitted_model.get_layer("ideal_points").get_weights()[0]
# m = asdf.transpose().dot(asdf)
# n = m - np.eye(2)
# rf = 0.1
# np.sqrt(rf * np.sum(np.square(np.abs(n))))
# np.sqrt(1e-8 * np.sum(np.square(np.abs(n))))
# np.sqrt(1.0 * np.sum(np.square(np.abs(n))))


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

n_users = vote_data["J"]
m_items = vote_data["M"]
k_dim = 1

# use_popularity = True
ideal_dropout = 0.0
yes_point_dropout = 0.0
no_point_dropout = 0.0
leg_input_dropout = 0.0
bill_input_dropout = 0.0

init_leg_embedding_final = pd.concat([vote_data["init_embedding"]["init_value"] * np.random.rand(vote_data["J"]) for j in range(k_dim)], axis=1)
if data_type == "test" and k_dim > 1:
    init_leg_embedding_final.iloc[:, 1] = 0
    init_leg_embedding_final.iloc[1, 1] = 0.1 * 1.0
# Make orthogonal
# init_leg_embedding_final.columns = range(k_dim)
# x -= x.dot(k) * k / np.linalg.norm(k)**2
# init_leg_embedding_final.corr()
# init_leg_embedding_final[1] -= (init_leg_embedding_final[1].dot(init_leg_embedding_final[0]) * init_leg_embedding_final[0] /
#                                 np.linalg.norm(init_leg_embedding_final[0])**2)
# init_leg_embedding_final.corr()
#
# init_pol_embedding_final = pd.concat([pd.DataFrame(np.random.randn(vote_data["M"]))for j in range(k_dim)], axis=1)
# # Make orthogonal
# init_pol_embedding_final.columns = range(k_dim)
# # x -= x.dot(k) * k / np.linalg.norm(k)**2
# init_pol_embedding_final.corr()
# init_pol_embedding_final[1] -= init_pol_embedding_final[1].dot(init_pol_embedding_final[0]) * init_pol_embedding_final[0] / np.linalg.norm(init_pol_embedding_final[0])**2
# init_pol_embedding_final.corr()

# vr = random_vector()
# vo = vr
# for v in (v1, v2, ... vn):
#     vo = vo - dot( vr, v ) / norm( v )
# if norm(vo) < k1 * norm(vr):
#     # this vector was mostly contained in the spanned subspace
# else:
#     # linearly independent, go ahead and use


def euc_dist_keras(x, y):
    # return K.sqrt(K.sum(K.square(x - y), axis=-1, keepdims=True))
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))


def wnom_term(tlist):
    w = tlist[0]
    x = tlist[1]
    z = tlist[2]
    temp_sum = K.dot(K.square(w), K.square(x - z))
    distances = K.sqrt(K.sum(temp_sum, axis=1, keepdims=True))
    return -0.5 * K.exp(distances)


class WnomTerm(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WnomTerm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[0][1]),
                                      initializer=Constant(0.5),  # match original
                                      trainable=True)
        super(WnomTerm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, tlist):
        x = tlist[0]
        z = tlist[1]
        # https://stackoverflow.com/questions/47289116/element-wise-multiplication-with-broadcasting-in-keras-custom-layer
        # x = K.print_tensor(x, message="x is: ")
        # z = K.print_tensor(z, message="z is: ")
        self.kernel = K.print_tensor(self.kernel, message="weights are: ")
        temp_sum = K.tf.multiply(K.square(x - z), K.square(self.kernel))
        temp_sum = K.print_tensor(temp_sum, message="temp_sum is: ")
        distances = K.sqrt(K.sum(temp_sum, axis=1, keepdims=True))
        distances = K.print_tensor(distances, message="distances is: ")
        result = K.exp(-0.5 * distances)
        result = K.print_tensor(result, message="distances is: ")
        return result

    def compute_output_shape(self, input_shape):
            return (input_shape[0][0], self.output_dim)


class JointWnomTerm(Layer):
    # TODO: Add unit norm constraint
    # TODO: What happens if I allow the weights ot be bill specific

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(JointWnomTerm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[0][1]),
                                      initializer=Constant(0.5),  # match original
                                      trainable=True)
        super(JointWnomTerm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, tlist):
        x = tlist[0]
        z1 = tlist[1]
        z2 = tlist[2]
        # self.kernel = K.print_tensor(self.kernel, message="weights are: ")
        # https://stackoverflow.com/questions/47289116/element-wise-multiplication-with-broadcasting-in-keras-custom-layer
        temp_sum1 = K.tf.multiply(K.square(x - z1), K.square(self.kernel))
        # temp_sum1 = K.tf.multiply(euc_dist_keras(x, z1), K.square(self.kernel))
        # temp_sum1 = K.tf.multiply(K.square(x - z1) + K.constant(1), K.square(self.kernel))
        # temp_sum1 = K.tf.multiply(K.square(x - z1) + K.epsilon(), K.square(self.kernel))
        # temp_sum1 = K.tf.multiply(K.maximum(K.square(x - z1), K.epsilon()), K.square(self.kernel))
        # temp_sum1 = K.tf.multiply(K.square(K.maximum(x - z1, K.epsilon())), K.square(self.kernel))
        # temp_sum1 = K.print_tensor(temp_sum1, message="temp_sum 1 is: ")
        # distances1 = K.sqrt(K.sum(temp_sum1, axis=1, keepdims=True))
        distances1 = K.sum(temp_sum1, axis=1, keepdims=True)
        # distances1 = K.sqrt(K.sum(temp_sum1, axis=1, keepdims=True) + K.epsilon())
        # distances1 = K.sqrt(K.maximum(K.sum(temp_sum1, axis=1, keepdims=True), K.constant(1e-9)))
        # distances1 = K.print_tensor(distances1, message="temp_sum 1 is: ")

        temp_sum2 = K.tf.multiply(K.square(x - z2), K.square(self.kernel))
        # temp_sum2 = K.tf.multiply(euc_dist_keras(x, z2), K.square(self.kernel))
        # temp_sum2 = K.tf.multiply(K.square(x - z2) + K.constant(1), K.square(self.kernel))
        # temp_sum2 = K.tf.multiply(K.square(x - z2) + K.epsilon(), K.square(self.kernel))
        # temp_sum2 = K.tf.multiply(K.maximum(K.square(x - z2), K.epsilon()), K.square(self.kernel))
        # temp_sum2 = K.tf.multiply(K.square(K.maximum(x - z2, K.epsilon())), K.square(self.kernel))
        # temp_sum2 = K.print_tensor(temp_sum2, message="temp_sum 2 is: ")
        # distances2 = K.sqrt(K.sum(temp_sum2, axis=1, keepdims=True))
        distances2 = K.sum(temp_sum2, axis=1, keepdims=True)
        # distances2 = K.sqrt(K.sum(temp_sum2, axis=1, keepdims=True) + K.epsilon())
        # distances2 = K.sqrt(K.maximum(K.sum(temp_sum2, axis=1, keepdims=True), K.constant(1e-9)))
        # distances2 = K.print_tensor(distances2, message="temp_sum 1 is: ")

        result = K.exp(-0.5 * distances1) - K.exp(-0.5 * distances2)
        # result = K.print_tensor(result, message="result is: ")
        return result

    def compute_output_shape(self, input_shape):
            return (input_shape[0][0], self.output_dim)


def standardize(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x /= K.std(x, axis=1)
    return x


def f(input_shape):
    return input_shape


def generate_time_input(i):
    time_input = Input(shape=(1, ), name="time_input_{}".format(i))
    return time_input


def normal_activation(x):
    dist = K.tf.distributions.Normal(loc=0.0, scale=1.0)
    return dist.cdf(x)

get_custom_objects().update({'normal_activation': Activation(normal_activation)})

# K.eval(dist.cdf(1.00))
# K.eval(K.sigmoid(1.00))

def generate_time_layer(i, leg_input, time_input):
    ideal_points_time = Embedding(input_dim=n_users, output_dim=k_dim, input_length=1, name="ideal_points_time_{}".format(i),
                                  embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                                  # weights=[init_leg_embedding_final.values]
                                  )(leg_input)
    ideal_points_time = Flatten()(ideal_points_time)
    if k_dim > 1:
        time_inputs_concat = Concatenate()([time_input] * k_dim)
    else:
        time_inputs_concat = time_input
    ideal_points_time = Multiply()([time_inputs_concat, ideal_points_time])
    return ideal_points_time


# Switch to the functional api (current version)
leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
if leg_input_dropout > 0.0:
    leg_input_drop = Dropout(leg_input_dropout)(leg_input)
else:
    leg_input_drop = leg_input
bill_input = Input(shape=(1, ), dtype="int32", name="bill_input")
if bill_input_dropout > 0.0:
    bill_input_drop = Dropout(bill_input_dropout)(bill_input)
else:
    bill_input_drop = bill_input
time_input_list = [generate_time_input(i) for i in range(1, k_time + 1)]
ideal_points = Embedding(input_dim=n_users, output_dim=k_dim, input_length=1, name="ideal_points",
                         # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                         # embeddings_regularizer=OrthReg(1e-1),
                         # embeddings_regularizer=UnitSphere(1e-1),
                         # embeddings_regularizer=regularizers.l2(1e-2),
                         # embeddings_constraint=unit_norm(axis=[0, 1]),
                         # embeddings_constraint=UnitSphere(),
                         embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                         weights=[init_leg_embedding_final.values],
                         )(leg_input_drop)
if ideal_dropout > 0.0:
    ideal_points = Dropout(ideal_dropout)(ideal_points)
# ideal_points = BatchNormalization()(ideal_points)
# ideal_points = Lambda(standardize, name="norm_ideal_points")(ideal_points)
time_layer_list = [generate_time_layer(i, leg_input_drop, time_input_list[i-1]) for i in range(1, k_time + 1)]

# flat_ideal_points = Reshape((k_dim,))(ideal_points)
# flat_ideal_points_time = Reshape((k_dim,))(ideal_points_time)

if k_time == 0:
    main_ideal_points = ideal_points
else:
    main_ideal_points = Add()([ideal_points] + time_layer_list)
# main_ideal_points = Lambda(standardize, name="norm_ideal_points")(main_ideal_points)
# main_ideal_points = BatchNormalization()(main_ideal_points)

flat_ideal_points = Reshape((k_dim,))(main_ideal_points)

yes_point = Embedding(input_dim=m_items, output_dim=k_dim, input_length=1, name="yes_point",
                      embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                      # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.3, seed=None),
                      )(bill_input_drop)

if yes_point_dropout > 0.0:
    yes_point = Dropout(yes_point_dropout)(yes_point)
flat_yes_point = Reshape((k_dim,))(yes_point)

no_point = Embedding(input_dim=m_items, output_dim=k_dim, input_length=1, name="no_point",
                     embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                     # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.3, seed=None),
                     )(bill_input_drop)
if no_point_dropout > 0.0:
    no_point = Dropout(no_point_dropout)(no_point)
flat_no_point = Reshape((k_dim,))(no_point)

# yes_term = WnomTerm(output_dim=1, trainable=True, name="yes_term")([flat_ideal_points, flat_yes_point])
# no_term = WnomTerm(output_dim=1, trainable=True, name="no_term")([flat_ideal_points, flat_no_point])
#
# combined = Subtract()([yes_term, no_term])

combined = JointWnomTerm(output_dim=1, trainable=True, name="wnom_term")([flat_ideal_points, flat_yes_point, flat_no_point])

# combined = K.print_tensor(combined, message="combined is: ")
main_output = Dense(1, activation="normal_activation", name="main_output", use_bias=False, kernel_initializer=Constant(15))(combined)

model = Model(inputs=[leg_input, bill_input] + time_input_list, outputs=[main_output])
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# model.compile(loss='mse', optimizer='adamax')
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

if data_type == "cosponsor":
    # base_weight = 1
    # upweight_yes = 100
    # sample_weights = [base_weight, base_weight * upweight_yes]
    sample_weights = (1.0 * vote_data["y"].shape[0]) / (len(np.unique(vote_data["y"])) * np.bincount(vote_data["y"]))
if data_type == "votes":
    sample_weights = (1.0 * vote_data["y"].shape[0]) / (len(np.unique(vote_data["y"])) * np.bincount(vote_data["y"]))
if data_type == "test":
    sample_weights = [1, 1]

# MODEL_WEIGHTS_FILE = ("~/temp/" + "collab_filter_example.h5")
# callbacks = [EarlyStopping('val_loss', patience=5),
#              ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
# history = model.fit([vote_data["j"], vote_data["m"]], vote_data["y"], epochs=100,
#                     validation_split=.2, verbose=2, callbacks=callbacks,
#                     class_weight={0: sample_weights[0],
#                                   1: sample_weights[1]})

callbacks = [EarlyStopping('val_loss', patience=20),
             GetBest(monitor='val_loss', verbose=1, mode='auto'),
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
ax = losses.iloc[1:, :].plot(x='epoch', figsize=[7, 10], grid=True)
ax.set_ylabel("accuracy")
ax.set_ylim([0.0, 3.0])

if data_type == "votes" or data_type == "cosponsor":
    leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

if data_type == "test":
    test_actual = False
    if test_actual:
        if k_dim == 1:
            leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0).reset_index()
        if k_dim == 2:
            leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0).reset_index()
    else:
        if k_dim == 1:
            leg_data = pd.read_csv(DATA_PATH + "/wnom1D_results.csv", index_col=0)
        if k_dim == 2:
            leg_data = pd.read_csv(DATA_PATH + "/wnom2D_results.csv", index_col=0)
        leg_data = leg_data.reset_index()

    leg_data = leg_data.rename(columns={"index": "leg_id",
                                        "icpsrState": "state_icpsr",
                                        "coord1D": "nominate_dim1",
                                        "coord2D": "nominate_dim2",
                                        "party.1": "party_code",
                                        "partyCode": "party_code"})
    leg_data["bioname"] = ""

    # fitted_model.get_layer("main_output").get_weights()[0]
    # fitted_model.get_layer("yes_term").get_weights()[0]
    # fitted_model.get_layer("no_term").get_weights()[0]
    # fitted_model.get_layer("yes_point").get_weights()[0]
    fitted_model.get_layer("main_output").get_weights()[0]
    fitted_model.layers[-2].get_weights()[0]
    # np.exp(0.5)
if k_dim == 1:
    leg_bio_data = leg_data[["leg_id", "state_icpsr", "bioname", "party_code", "nominate_dim1"]].drop_duplicates()
if k_dim > 1:
    leg_bio_data = leg_data[["leg_id", "state_icpsr", "bioname", "party_code", "nominate_dim1", "nominate_dim2"]].drop_duplicates()

ideal_point_names = ["ideal_{}".format(j) for j in range(1, k_dim + 1)]
drift_names = ["drift_{}".format(j) for j in range(1, k_time + 1)]
yes_point_names = ["yes_point_{}".format(j) for j in range(1, k_dim + 1)]
no_point_names = ["no_point_{}".format(j) for j in range(1, k_dim + 1)]
if k_dim == 1:
    var_list = ["nominate_dim1"] + ideal_point_names + drift_names
if k_dim > 1:
    var_list = ["nominate_dim1", "nominate_dim2"] + ideal_point_names + drift_names

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

leg_data_cf.groupby("party_code").mean()


leg_data_cf
A = leg_data_cf.dropna()
x = np.linalg.lstsq(A[ideal_point_names].values, A[["nominate_dim1", "nominate_dim2"]].values)[0]

y_hat = A[ideal_point_names].values.dot(x)

pd.concat([A[["nominate_dim1", "nominate_dim2"]], pd.DataFrame(y_hat, index=A.index)], axis=1).corr()
