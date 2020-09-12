import os
import numpy as np
import pandas as pd

import pickle

import warnings

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras.layers import Embedding, Reshape, Dropout, SpatialDropout1D, Dense, Flatten, Input, Dot, LSTM, Add, Subtract, Conv1D, MaxPooling1D, Concatenate, Multiply, BatchNormalization, Lambda, Activation, InputSpec
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.constraints import Constraint, unit_norm, MinMaxNorm, non_neg
# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras import optimizers
from tensorflow.keras.utils import get_custom_objects
# from tensorflow.keras.layers import constraints
from tensorflow.keras import constraints

from tensorflow.keras.models import load_model

from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot

from tensorflow.keras import backend as K

import tensorflow.keras.backend as B
# import tensorflow_hub as hub
# from tensorflow.python.keras.engine import Layer


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
            # filepath = self.filepath.format(epoch=epoch + 1, **logs)
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

    def __init__(self, rf=1.0, axis=0):
        self.rf = K.cast_to_floatx(rf)
        self.axis = axis

    def __call__(self, xy_t):
        regularization = 0.0
        mean_t = K.mean(xy_t, axis=self.axis, keepdims=True)
        dsize = K.int_shape(mean_t)[1]
        if dsize > 1:
            cov_t = (K.transpose(xy_t-mean_t) @ (xy_t-mean_t)) / (dsize - 1)
            cov2_t = tf.linalg.diag(1 / K.sqrt(tf.linalg.diag_part(cov_t) + K.epsilon()))
            cor = cov2_t @ cov_t @ cov2_t
            # Norm of the off diagonal elements
            eye = np.eye(K.int_shape(cor)[0])
            # Penalize by the off diagonal elements
            norm = self.rf * K.sqrt(K.sum(K.square(cor - eye)) + K.epsilon())
            # norm = K.print_tensor(norm, message="norm is: ")
        else:
            norm = 0
        regularization += norm

        # QA help
        # mean_t = K.mean(xy_t, axis=0, keepdims=True)
        # K.eval(mean_t)
        # cov_t = (K.transpose(xy_t-mean_t) @ (xy_t-mean_t)) / (dsize - 1)
        # K.eval(cov_t)
        # cov2_t = tf.diag(1 / K.sqrt(tf.diag_part(cov_t)))
        # K.eval(cov2_t)
        # cor = cov2_t @ cov_t @ cov2_t
        # K.eval(cor)
        # K.eval(K.sqrt(K.sum(K.square(cor - eye)) + K.epsilon()))

        return regularization

    def get_config(self):
        return {'rf': float(self.rf),
                'axis': self.axis}


class TimestepDropout(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - https://github.com/keras-team/keras/issues/7290
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape


class UnitMetric(Constraint):
    """UnitMetric weight constraint.
    Constrains the weights incident to each hidden unit
    to have a norm ranging between 0 and 1.
    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        # w = K.random_uniform_variable(shape=(5, 2), low=-0.5, high=0.5)
        # norms = K.sqrt(K.sum(K.square(w), axis=1, keepdims=True))
        # K.eval(w)
        # K.eval(norms)
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        max_norm = K.max(norms)
        # new_w = w / (max_norm + K.epsilon())
        # K.eval(new_w)
        # K.eval(K.sqrt(K.sum(K.square(new_w), axis=1, keepdims=True)))
        w /= (max_norm + K.epsilon())
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}


class UnitSphere(Constraint):
    """Constrain to unit hypersphere

    # Arguments
        rf: Float; rf regularization factor.
    """
    def __call__(self, w):
        temp_sum = K.sum(K.square(w), axis=0)
        w *= K.cast(K.less_equal(temp_sum, 1.0), K.floatx())
        return w


class SumToOne(Constraint):
    """Sum to one constraint
    """
    def __call__(self, w):
        # Guarantee non-negativity
        # Straight from the NonNeg code
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        # Renormalize
        w /= K.sum(w, axis=-1, keepdims=True)
        return w


class JointWnomTerm(Layer):
    # TODO: Add unit norm constraint
    # IDEA: What happens if I allow the weights to be bill specific (probably not a good idea)

    def __init__(self, output_dim, kernel_constraint=None, kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_regularizer = constraints.get(kernel_regularizer)
        super(JointWnomTerm, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[0][1]),
                                      initializer=Constant(0.5),  # match original
                                      constraint=self.kernel_constraint,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(JointWnomTerm, self).build(input_shape)  # Be sure to call this at the end

    def call(self, tlist):
        x = tlist[0]
        z1 = tlist[1]
        z2 = tlist[2]
        # self.kernel = K.print_tensor(self.kernel, message="weights are: ")
        # https://stackoverflow.com/questions/47289116/element-wise-multiplication-with-broadcasting-in-keras-custom-layer
        # temp_sum1 = tf.multiply(K.square(x - z1), K.square(self.kernel))
        # distances1 = K.sum(temp_sum1, axis=1, keepdims=True)
        distances1 = K.dot(K.square(x - z1), K.transpose(K.square(self.kernel)))
        # distances1 = K.print_tensor(distances1, message="result is: ")

        # temp_sum2 = tf.multiply(K.square(x - z2), K.square(self.kernel))
        # distances2 = K.sum(temp_sum2, axis=1, keepdims=True)
        distances2 = K.dot(K.square(x - z2), K.transpose(K.square(self.kernel)))

        result = K.exp(-0.5 * distances1) - K.exp(-0.5 * distances2)
        # result = K.print_tensor(result, message="result is: ")
        return result

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            # 'units': self.units,
            # 'activation': activations.serialize(self.activation),
            # 'use_bias': self.use_bias,
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            # 'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            # 'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            # 'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(JointWnomTerm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)


def standardize(x):
    """Standardize a tensor to standard normal
    """
    x -= K.mean(x, axis=1, keepdims=True)
    x /= K.std(x, axis=1)
    return x


def generate_time_input(i):
    """Initialize a time input layer
    """
    time_input = Input(shape=(1, ), name="time_input_{}".format(i))
    return time_input


def generate_time_layer(i, n_leg, k_dim, leg_input, time_input):
    ideal_points_time = Embedding(input_dim=n_leg, output_dim=k_dim, input_length=1, name="ideal_points_time_{}".format(i),
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


def normal_activation(x):
    """Use a standard normal cdf as the activation function
    """
    dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    get_custom_objects().update({'normal_activation': Activation(normal_activation)})
    return dist.cdf(x)


# Register the new activation function prior to use
get_custom_objects().update({'normal_activation': Activation(normal_activation)})


def NNnominate(n_leg, n_votes,
               k_dim=2,
               k_time=0,
               init_leg_embedding=None,
               ideal_dropout=0.0,
               yes_point_dropout=0.0,
               no_point_dropout=0.0,
               combined_dropout=0.0,
               dropout_type="timestep",
               covariates_list=[],
               k_out=1,
               main_activation='sigmoid',
               ):
    """A function to build a dwnominate style neural network

    # Arguments:
        n_leg (int): the number of legislators in the model
        n_votes (int): the number of votes in the model
        k_dim (int): the number of dimensions in the model
        k_time (int), EXPERIMENTAL: the number of time dimensions in the model,
            if k_time > 1 a legislator's ideal point at time t is implemented as
            a polynomial function of degree k_time. Note that the current
            implementation may result in unexpected behavior (e.g. an ideal
            point outside the unit sphere)
        init_leg_embedding (pd.Dataframe): initial values for the legislator
            embeddings of shape n_leg x k_dim
        ideal_dropout (float): ideal point dropout rate
        yes_point_dropout (float): yes point dropout rate
        no_point_dropout (float): no point dropout rate
        dropout_type (str): if timestep, an entire bill/legislator will be
            dropped at random, otherwise, a constant fraction of all weights
            will be dropped
        covariates_list (list), EXPERIMENTAL: a list of covariate names to
            initialize addition of covariates to the model
        main_activation (str), EXPERIMENTAL: activation function to use for
            main layer, possible values ["sigmoid", "guassian"]
    # Returns:
        A keras model ready for compilation and fit
    """
    # Set up inputs for embedding layers, lists of integer ids
    leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
    bill_input = Input(shape=(1, ), dtype="int32", name="bill_input")

    # Generate inputs
    time_input_list = [generate_time_input(i) for i in range(1, k_time + 1)]

    # If initial weights are not provided, set at random
    if init_leg_embedding.empty:
        init_leg_embedding = pd.DataFrame(np.random.uniform(-1, 1, size=(n_leg, k_dim)))

    # Set up the ideal points embedding layer
    # OrthReg ensures that the dimensions of the ideal point vector are uncorrelated
    # MinMaxNorm ensures that all ideal points lie within the unit sphere; note that
    # this is not the same as ensuring that any ideal points reach the edge of the
    # unit sphere which is a slight difference from the conventional results.
    ideal_points = Embedding(input_dim=n_leg, output_dim=k_dim, input_length=1, name="ideal_points",
                             # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                             embeddings_regularizer=OrthReg(1e-1),
                             # embeddings_regularizer=UnitSphere(1e-1),
                             # embeddings_regularizer=regularizers.l2(1e-2),
                             # embeddings_constraint=unit_norm(),
                             # embeddings_constraint=UnitSphere(),
                             # embeddings_constraint=UnitMetric(axis=1),
                             # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                             weights=[init_leg_embedding.values],
                             )(leg_input)
    # Dropout regularization of the ideal points
    if ideal_dropout > 0.0:
        if dropout_type == "timestep":
            ideal_points = TimestepDropout(ideal_dropout)(ideal_points)
        else:
            ideal_points = Dropout(ideal_dropout)(ideal_points)
    # Normalize the layers (not strictly necessary)
    # ideal_points = BatchNormalization()(ideal_points)
    # ideal_points = Lambda(standardize, name="norm_ideal_points")(ideal_points)

    # Obtain layers for time (if any)
    time_layer_list = [generate_time_layer(i, n_leg, k_dim, leg_input, time_input_list[i-1]) for i in range(1, k_time + 1)]

    # Join the time layers with the embedding layer
    if k_time == 0:
        main_ideal_points = ideal_points
    else:
        main_ideal_points = Add()([ideal_points] + time_layer_list)
    # main_ideal_points = BatchNormalization()(main_ideal_points)
    # main_ideal_points = Lambda(standardize, name="norm_ideal_points")(main_ideal_points)

    # Reshape to drop unecessary dimensions left from embeddings
    flat_ideal_points = Reshape((k_dim,))(main_ideal_points)

    # Generate yes_point embedding layer
    yes_point = Embedding(input_dim=n_votes, output_dim=k_dim, input_length=1, name="yes_point",
                          # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                          embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.3, seed=None),
                          embeddings_regularizer=OrthReg(1e-1),
                          )(bill_input)
    # yes_point dropout regularization
    if yes_point_dropout > 0.0:
        if dropout_type == "timestep":
            yes_point = TimestepDropout(yes_point_dropout, seed=65)(yes_point)
        else:
            yes_point = Dropout(yes_point_dropout, seed=65)(yes_point)
    # Reshape to drop unnecessary dimensions left from embeddings
    flat_yes_point = Reshape((k_dim,))(yes_point)

    # Generate no_point embedding layer
    no_point = Embedding(input_dim=n_votes, output_dim=k_dim, input_length=1, name="no_point",
                         # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=1.0, axis=1, rate=1.0),
                         embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.3, seed=None),
                         embeddings_regularizer=OrthReg(1e-1),
                         )(bill_input)
    # no_point dropout regularization
    if no_point_dropout > 0.0:
        if dropout_type == "timestep":
            no_point = TimestepDropout(no_point_dropout, seed=65)(no_point)
        else:
            no_point = Dropout(no_point_dropout, seed=65)(no_point)
    # Reshape to drop unnecessary dimensions left from embeddings
    flat_no_point = Reshape((k_dim,))(no_point)

    # Combine ideal_points, yes_points, and no_points using a custom dwnominate layer
    combined = JointWnomTerm(output_dim=1, trainable=True, name="wnom_term",
                             kernel_constraint=non_neg(),
                             # kernel_constraint=SumToOne(),
                             # kernel_regularizer=regularizers.l2(),
                             )([flat_ideal_points, flat_yes_point, flat_no_point])
    # Combined dropout regularization
    # Setting this sets salience weights for a random dimension to 0
    if combined_dropout > 0:
        combined = Dropout(combined_dropout)(combined)

    # Include the covariates (if any)
    if covariates_list:
        print(covariates_list)
        covariates = Input(shape=(len(covariates_list), ), name="covariates")
        combined = Concatenate()([combined, covariates])

    # Final output, a simple logistic layer
    if main_activation == "sigmoid":
        main_output = Dense(k_out, activation="sigmoid", name="main_output", use_bias=False, kernel_initializer=Constant(1))(combined)
    elif main_activation == "gaussian":
        main_output = Dense(k_out, activation=normal_activation, name="main_output", use_bias=False, kernel_initializer=Constant(1))(combined)

    # Define model, depending on existence of covariates and time elements
    if covariates_list:
        if k_time > 0:
            model = Model(inputs=[leg_input, bill_input] + time_input_list + [covariates], outputs=[main_output])
        else:
            model = Model(inputs=[leg_input, bill_input] + [covariates], outputs=[main_output])
    else:
        if k_time > 0:
            model = Model(inputs=[leg_input, bill_input] + time_input_list, outputs=[main_output])
        else:
            model = Model(inputs=[leg_input, bill_input], outputs=[main_output])
    return model


def NNitemresponse(n_leg, n_votes,
                   k_dim=2,
                   k_time=0,
                   init_leg_embedding=None,
                   ideal_dropout=0.0,
                   polarity_dropout=0.0,
                   use_popularity=True,
                   popularity_dropout=0.0,
                   combined_dropout=0.0,
                   dropout_type="timestep",
                   covariates_list=[],
                   k_out=1,
                   batch_normalize=False,
                   ):
    """A function to build a dwnominate style neural network

    # Arguments:
        n_leg (int): the number of legislators in the model
        n_votes (int): the number of votes in the model
        k_dim (int): the number of dimensions in the model
        k_time (int), EXPERIMENTAL: the number of time dimensions in the model,
            if k_time > 1 a legislator's ideal point at time t is implemented as
            a polynomial function of degree k_time. Note that the current
            implementation may result in unexpected behavior (e.g. an ideal
            point outside the unit sphere)
        init_leg_embedding (pd.Dataframe): initial values for the legislator
            embeddings of shape n_leg x k_dim
        ideal_dropout (float): ideal point dropout rate
        polarity_dropout (float): polarity dropout rate
        use_popularity (bool): include a popularity bill parameter
        popularity_dropout (float): popularity dropout rate
        dropout_type (str): if timestep, an entire bill/legislator will be
            dropped at random, otherwise, a constant fraction of all weights
            will be dropped
        covariates_list (list), EXPERIMENTAL: a list of covariate names to
            initialize addition of covariates to the model
    # Returns:
        A keras model ready for compilation and fit
    """
    # Set up inputs for embedding layers, lists of integer ids
    leg_input = Input(shape=(1, ), dtype="int32", name="leg_input")
    bill_input = Input(shape=(1, ), dtype="int32", name="bill_input")

    # Generate inputs
    time_input_list = [generate_time_input(i) for i in range(1, k_time + 1)]

    # If initial weights are not provided, set at random
    if init_leg_embedding.empty:
        init_leg_embedding = pd.DataFrame(np.random.uniform(-1, 1, size=(n_leg, k_dim)))

    # Set up the ideal points embedding layer
    # OrthReg ensures that the dimensions of the ideal point vector are uncorrelated
    # MinMaxNorm ensures that all ideal points lie within the unit sphere; note that
    # this is not the same as ensuring that any ideal points reach the edge of the
    # unit sphere which is a slight difference from the conventional results.
    ideal_points = Embedding(input_dim=n_leg, output_dim=k_dim, input_length=1, name="ideal_points",
                             # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                             embeddings_regularizer=OrthReg(1e-1),
                             # activity_regularizer=regularizers.l2(1e-6),
                             # embeddings_regularizer=UnitSphere(1e-1),
                             # embeddings_regularizer=regularizers.l2(1e-5),
                             # embeddings_constraint=unit_norm(axis=1),
                             # embeddings_constraint=UnitSphere(),
                             # embeddings_constraint=UnitMetric(axis=1),
                             # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=3.0, axis=1, rate=1.0),
                             weights=[init_leg_embedding.values],
                             )(leg_input)
    # Dropout regularization of the ideal points
    if ideal_dropout > 0.0:
        if dropout_type == "timestep":
            ideal_points = TimestepDropout(ideal_dropout)(ideal_points)
        else:
            ideal_points = Dropout(ideal_dropout)(ideal_points)

    if batch_normalize:
        main_ideal_points = BatchNormalization(name="norm_ideal_points")(ideal_points)
    else:
        main_ideal_points = ideal_points

    # Reshape to drop unecessary dimensions left from embeddings
    flat_ideal_points = Reshape((k_dim,))(main_ideal_points)

    polarity = Embedding(input_dim=n_votes, output_dim=k_dim, input_length=1, name="polarity",
                         # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=3.0, axis=1, rate=1.0),
                         # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                         )(bill_input)
    if polarity_dropout > 0.0:
        polarity = Dropout(polarity_dropout)(polarity)
    flat_polarity = Reshape((k_dim,))(polarity)
    if use_popularity:
        popularity = Embedding(input_dim=n_votes, output_dim=1, input_length=1, name="popularity",
                               # embeddings_constraint=MinMaxNorm(min_value=0.0, max_value=3.0, axis=1, rate=1.0),
                               # embeddings_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                               )(bill_input)
        if popularity_dropout > 0.0:
            popularity = Dropout(popularity_dropout)(popularity)
        flat_popularity = Flatten()(popularity)
        combined_temp = Dot(axes=1)([flat_ideal_points, flat_polarity])
        combined = Add()([combined_temp, flat_popularity])
    else:
        combined = Dot(axes=1)([flat_ideal_points, flat_polarity])

    # Combined dropout regularization
    # Setting this sets salience weights for a random dimension to 0
    if combined_dropout > 0:
        combined = Dropout(combined_dropout)(combined)

    # Include the covariates (if any)
    if covariates_list:
        print(covariates_list)
        covariates = Input(shape=(len(covariates_list), ), name="covariates")
        combined = Concatenate()([combined, covariates])

    main_output = Dense(1, activation="sigmoid", name="main_output", use_bias=False, kernel_initializer=Constant(1.0), trainable=False)(combined)

    model = Model(inputs=[leg_input, bill_input], outputs=[main_output])

    # Define model, depending on existence of covariates and time elements
    if covariates_list:
        if k_time > 0:
            model = Model(inputs=[leg_input, bill_input] + time_input_list + [covariates], outputs=[main_output])
        else:
            model = Model(inputs=[leg_input, bill_input] + [covariates], outputs=[main_output])
    else:
        if k_time > 0:
            model = Model(inputs=[leg_input, bill_input] + time_input_list, outputs=[main_output])
        else:
            model = Model(inputs=[leg_input, bill_input], outputs=[main_output])
    return model


if __name__ == '__main__':
    fsize=1
    dsize=5
    x=np.random.random((fsize,dsize))
    y=np.random.random((fsize,dsize))
    xy=np.concatenate([x,y], axis=0)

    x
    y
    xy

    xy_t = K.random_normal_variable(shape=(5, 1), mean=0, scale=1)
    K.eval(xy_t)

    mean_t = K.mean(xy_t, axis=0, keepdims=True)
    K.eval(mean_t)
    cov_t = (K.transpose(xy_t-mean_t) @ (xy_t-mean_t)) / (dsize - 1)
    K.eval(cov_t)
    cov2_t = tf.linalg.diag(1 / K.sqrt(tf.linalg.diag_part(cov_t)))
    K.eval(cov2_t)
    cor = cov2_t @ cov_t @ cov2_t

    xy = K.eval(xy_t)
    np.testing.assert_allclose(np.corrcoef(xy, rowvar=False), K.eval(cor), rtol=1e-06)

    K
