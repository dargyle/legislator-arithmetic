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

tfd = tfp.distributions


vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

print("Get nn_estimate")
# Process the vote_df into the model data format
i = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=0,
               k_dim=i,
               k_time=0,
               covariates_list=[],
               )
vote_data = process_data(**data_params)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)


"""
Probabilities for a single person
discriminations   I (item)
difficulties I x K-1
abilities N

returns
prob N x I x K
"""

@tf.function
def grm_model_prob(abilities, discriminations, difficulties, expanded=False):
    offsets = difficulties[tf.newaxis, :, :] - abilities[:, tf.newaxis, tf.newaxis]
    scaled = offsets*discriminations[tf.newaxis, :, tf.newaxis]
    logits = 1.0/(1+tf.exp(scaled))
    logits = tf.pad(logits, paddings=(
        (0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1.)
    logits = tf.pad(logits, paddings=(
        (0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0.)
    probs = logits[:, :, :-1] - logits[:, :, 1:]

    return probs

"""
Probabilities for single items
discriminations   I (item)
difficulties  I x K - 1
abilities N

mu (difficulty local) I
tau (difficulty) I x K-2

"""
@tf.function
def joint_log_prob(responses, discriminations, difficulties0, ddifficulties, abilities, mu):
    # assemble the difficulties
    d0 = tf.concat([difficulties0[:,tf.newaxis], ddifficulties],axis=1)
    difficulties = tf.cumsum(d0, axis=1)
    return tf.reduce_sum(log_likelihood(responses, discriminations, difficulties, abilities)) + \
        joint_log_prior(discriminations, difficulties0, ddifficulties, abilities, mu)


@tf.function
def log_likelihood(responses, discriminations, difficulties, abilities):
    rv_responses = tfd.Categorical(grm_model_prob(
        abilities, discriminations, difficulties))
    return rv_responses.log_prob(responses)


@tf.function
def joint_log_prior(discriminations, difficulties0, ddifficulties, abilities, mu):
    rv_discriminations = tfd.HalfNormal(scale=tf.ones_like(discriminations))
    rv_difficulties0 = tfd.Normal(loc=mu, scale=1.*tf.ones_like(difficulties0))
    rv_ddifficulties = tfd.HalfNormal(scale=tf.ones_like(ddifficulties))
    rv_abilities = tfd.Normal(loc=tf.zeros_like(abilities), scale=1.)
    rv_mu = tfd.Normal(loc=tf.zeros_like(mu), scale=1.)

    return tf.reduce_sum(rv_discriminations.log_prob(discriminations)) + \
        tf.reduce_sum(rv_difficulties0.log_prob(difficulties0)) + \
        tf.reduce_sum(rv_ddifficulties.log_prob(ddifficulties)) + \
        tf.reduce_sum(rv_abilities.log_prob(abilities))


difficulties0 = np.sort(np.random.normal(size=(I, K-1)), axis=1)
abilities0 = np.random.normal(size=N)

# Set the chain's start state.
initial_chain_state = [
    tf.ones((I), name='init_discriminations'),
    tf.cast(difficulties0[:, 0], tf.float32, name='init_difficulties0'), # may be causing problems
    tf.cast(difficulties0[:, 1:]-difficulties0[:, :-1], tf.float32, name='init_ddifficulties'),
    tf.cast(abilities0, tf.float32, name='init_abilities'),
    tf.zeros((I), name='init_mu')
]

# Since MCMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Softplus(),       # R^+ \to R
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Softplus(),
    tfp.bijectors.Identity(),       # Maps R to R.
    tfp.bijectors.Identity()       # Maps R to R.
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = tf.function(lambda *args: joint_log_prob(data, *args))

# Initialize the step_size. (It will be automatically adapted.)


number_of_steps=10000
burnin=6000
num_leapfrog_steps=4
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=0.1,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

mh = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=unnormalized_posterior_log_prob),
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))

def trace_everything(states, previous_kernel_results):
  return previous_kernel_results

[discriminations,difficulties0,ddifficulties,abilities,mu], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    trace_fn=trace_everything,
    kernel=kernel)

with tf.device('/device:GPU:0'):
  [
    discriminations_,
    difficulties0_,
    ddifficulties_,
    abilities_,mu_,
    kernel_results_
  ] = evaluate(
      [discriminations,
       difficulties0,
       ddifficulties,
       abilities,
       mu,
       kernel_results
      ])




# model = tfp.experimental.inference_gym.targets.ItemResponseTheory(x_train[0], x_train[1], vote_data["y_train"], x_test[0], x_test[1], vote_data["y_test"])
# model.__dict__
# tf.zeros(model.event_shape["student_ability"])
#
# samples = tfp.mcmc.HamiltonianMonteCarlo(model.unnormalized_log_prob, step_size=1, num_leapfrog_steps=10)
#
# num_results = int(10e3)
# num_burnin_steps = int(1e3)
# adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#     tfp.mcmc.HamiltonianMonteCarlo(model.unnormalized_log_prob, step_size=1.0, num_leapfrog_steps=3),
#     num_adaptation_steps=int(num_burnin_steps * 0.8),
#     )
#
# adaptive_hmc
#
#
# hmc = tfp.mcmc.TransformedTransitionKernel(
#     inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
#         target_log_prob_fn=model.unnormalized_log_prob, step_size=1.0, num_leapfrog_steps=3
#         ),
#     bijector=model.default_event_space_bijector,
#     )
#
# tfp.mcmc.sample_chain(num_results=10000, current_state=[tf.ones(model.event_shape["student_ability"]), tf.ones(model.event_shape["question_difficulty"])], kernel=hmc)
#
# @tf.function
# def run_chain():
#     # Run the chain (with burn-in).
#     samples, is_accepted = tfp.mcmc.sample_chain(
#         num_results=num_results,
#         num_burnin_steps=num_burnin_steps,
#         current_state=[tf.ones(model.event_shape["student_ability"]), tf.ones(model.event_shape["question_difficulty"])],
#         kernel=adaptive_hmc,
#         trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
#     sample_mean = tf.reduce_mean(samples)
#     sample_stddev = tf.math.reduce_std(samples)
#     is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
#     return sample_mean, sample_stddev, is_accepted
#
#
# sample_mean, sample_stddev, is_accepted = run_chain()
#
#
#   # Target distribution is proportional to: `exp(-x (1 + x))`.
#   def unnormalized_log_prob(x):
#     return -x - x**2.
#   # Initialize the HMC transition kernel.
#   num_results = int(10e3)
#   num_burnin_steps = int(1e3)
#   adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#       tfp.mcmc.HamiltonianMonteCarlo(
#           target_log_prob_fn=unnormalized_log_prob,
#           num_leapfrog_steps=3,
#           step_size=1.),
#       num_adaptation_steps=int(num_burnin_steps * 0.8))
#   # Run the chain (with burn-in).
#   @tf.function
#   def run_chain():
#     # Run the chain (with burn-in).
#     samples, is_accepted = tfp.mcmc.sample_chain(
#         num_results=num_results,
#         num_burnin_steps=num_burnin_steps,
#         current_state=1.,
#         kernel=adaptive_hmc,
#         trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
#     sample_mean = tf.reduce_mean(samples)
#     sample_stddev = tf.math.reduce_std(samples)
#     is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
#     return sample_mean, sample_stddev, is_accepted
#   sample_mean, sample_stddev, is_accepted = run_chain()
