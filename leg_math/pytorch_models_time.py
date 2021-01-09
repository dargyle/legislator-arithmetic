import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides

from pyro.infer.autoguide.initialization import init_to_value

import pandas as pd
import numpy as np

from data_generation.data_processing import process_data, format_model_data
from data_generation.random_votes import generate_nominate_votes

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

pyro.enable_validation(True)


# Set up environment
# Untested with GPU, that's part of what this is about
gpu = False

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Add a covariate
random_votes = generate_nominate_votes(n_leg=25, n_votes=100, beta=15.0, beta_covar=5.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
vote_df = random_votes.reset_index()

k_dim = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
               k_dim=k_dim,
               k_time=2,
               covariates_list=["in_majority"],
               unanimity_check=False,
               )
vote_data = process_data(**data_params)
custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

# Convert training and test data to tensors
legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
time_passed = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# Set some constants
num_legs = len(set(legs.numpy()))
num_votes = len(set(votes.numpy()))
num_covar = covariates.shape[1]


def ideal_point_model_covar(legs, votes, y=None, covariates=None, time_passed=None, k_dim=1):
    """Define a core ideal point model

    Args:
        legs: a tensor of legislator ids
        votes: a tensor of vote ids
        y: a tensor of vote choices
        k_dim: desired dimensions of the models
    """
    # Set up parameter plates for all of the parameters
    if time_passed is not None:
        k_time = time_passed.shape[1]
        with pyro.plate('thetas', num_legs, dim=-3, device=device):
            ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, k_time, device=device), torch.ones(k_dim, k_time, device=device)))
        final_ideal_point = torch.sum(ideal_point[legs] * time_passed[votes].unsqueeze(dim=1), axis=2)
        # ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1)
        # ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1)
        # torch.sum(ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1), axis=1)
    else:
        with pyro.plate('thetas', num_legs, dim=-2, device=device):
            final_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', num_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', num_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    if covariates is not None:
        with pyro.plate('coefs', num_covar, device=device):
            coef = pyro.sample('coef', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))
        # print(covariates.shape)
        # print(coef.unsqueeze(-1).shape)
        covar_combo = torch.mm(covariates, coef.unsqueeze(-1)).flatten()
        # print(covar_combo.shape)
        logit = torch.sum(final_ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes] + covar_combo
    else:
        logit = torch.sum(final_ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes]
    # Used for debugging
    # print('ideal_shape: {}'.format(ideal_point[legs].shape))
    # print('polarity_shape: {}'.format(polarity[votes].shape))
    # print('popularity_shape: {}'.format(popularity[votes].shape))
    # print(ideal_point[legs])

    # Combine parameters
    # logit = torch.sum(ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes] + covar_combo
    # print(logit)
    # print(logit.shape)

    if y is not None:
        # If training outcomes are provided, run the sampling statement for the given data
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        # If no training outcomes are provided, return the samples from the predicted distributions
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


optim = Adam({'lr': 0.1})

# Define the guide
guide = autoguides.AutoNormal(ideal_point_model_covar)
# guide = autoguides.AutoNormal(ideal_point_model, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
# guide = ideal_point_guide(legs, votes, responses, i)

# Setup the variational inference
svi = SVI(ideal_point_model_covar, guide, optim, loss=Trace_ELBO())
svi.step(legs, votes, responses, covariates, time_passed, k_dim=k_dim)
# svi.step(torch.tensor([1,1]), torch.tensor([0,1]), torch.tensor([0.,1.]), k_dim=k_dim)

# Run variational inference
pyro.clear_param_store()
for j in range(2000):
    loss = svi.step(legs, votes, responses, covariates, time_passed, k_dim=k_dim)
    if j % 100 == 0:
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))

pyro.param("AutoNormal.locs.theta").data.numpy()
pyro.param("AutoNormal.locs.coef").data.numpy()
pyro.param("AutoNormal.scales.coef").data.numpy()

init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                    "beta": pyro.param("AutoNormal.locs.beta").data,
                                    "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                    })

# Set up sampling alrgorithm
nuts_kernel = NUTS(ideal_point_model_covar, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
# For real inference should probably increase the number of samples, but this is slow and enough to test
hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
# Run the model
hmc_posterior.run(legs, votes, responses, covariates, time_passed, k_dim=k_dim)

samples = hmc_posterior.get_samples()
pd.DataFrame(samples["coef"].numpy()).hist()

samples["theta"].mean(axis=0).shape

samples["theta"].mean(axis=0)[5]
