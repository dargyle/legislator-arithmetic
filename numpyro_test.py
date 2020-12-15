import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS, SA, Predictive, log_likelihood

import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp

import pandas as pd
from data_generation.data_processing import process_data, format_model_data

from constants import DATA_PATH

# gpu = False
#
# device = torch.device('cpu')
# if gpu:
#     device = torch.device('cuda')

vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

i = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=116,
               k_dim=i,
               k_time=1,
               covariates_list=[],
               unanimity_check=False,
               )
vote_data = process_data(**data_params)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

legs = x_train[0].flatten()
votes = x_train[1].flatten()
responses = vote_data["y_train"].flatten()

num_legs = len(set(legs))
num_votes = len(set(votes))
print(num_votes, num_legs)

legs = jnp.array(legs)
votes = jnp.array(votes)
responses = jnp.array(responses)


def temp_model(legs, votes, obs, dim=1):
    with numpyro.plate('thetas', num_legs, dim=-2):
        ideal_point = numpyro.sample('theta', dist.Normal(jnp.zeros(dim), jnp.ones(dim)))

    with numpyro.plate('bs', num_votes, dim=-2,  device=device):
        polarity = numpyro.sample('b', dist.Normal(jnp.zeros(dim), 1.e3 * jnp.ones(dim)))

    with numpyro.plate('cs', num_votes):
        popularity = numpyro.sample('c', dist.Normal(jnp.zeros(1), 1.e3 * jnp.ones(1)))

    # print('ideal_shape: {}'.format(ideal_point[legs].shape))
    # print('polarity_shape: {}'.format(polarity[votes].shape))
    # print('popularity_shape: {}'.format(popularity[votes].shape))

    logit = jnp.sum(ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes]
    # print(logit)
    # print(logit.shape)

    with numpyro.plate('observe_data', obs.size(0)):
        numpyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)


nuts_kernel = NUTS(temp_model, adapt_step_size=True)
hmc_posterior = MCMC(nuts_kernel, 100, 100)
hmc_posterior.run(legs, votes, responses, dim=2)
