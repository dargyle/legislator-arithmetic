import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides

from pyro.infer.autoguide.initialization import init_to_value

import pandas as pd

from data_generation.data_processing import process_data, format_model_data

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

# Read in the US data and subset to a simple test dataset (US Senate)
vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

# Single dimension for now
k_dim = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=116,
               k_dim=k_dim,
               k_time=0,
               covariates_list=[],
               unanimity_check=False,
               )
vote_data = process_data(**data_params)
custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

# Convert training and test data to tensors
legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)

# Set some constants
num_legs = len(set(legs.numpy()))
num_votes = len(set(votes.numpy()))


def ideal_point_model(legs, votes, y=None, k_dim=1):
    """Define a core ideal point model

    Args:
        legs: a tensor of legislator ids
        votes: a tensor of vote ids
        y: a tensor of vote choices
        k_dim: desired dimensions of the models
    """
    # Set up parameter plates for all of the parameters
    with pyro.plate('thetas', num_legs, dim=-2, device=device):
        ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', num_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', num_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    # Used for debugging
    # print('ideal_shape: {}'.format(ideal_point[legs].shape))
    # print('polarity_shape: {}'.format(polarity[votes].shape))
    # print('popularity_shape: {}'.format(popularity[votes].shape))
    # print(ideal_point[legs])

    # Combine parameters
    logit = torch.sum(ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes]
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


def ideal_point_guide(legs, votes, obs, k_dim=1):
    """ Work in progress to set up a custom guide, for now relty on autoguide below
    """
    # register learnable params in the param store
    # m_theta_param = pyro.param("loc_ability", torch.zeros(num_legs, device=device))
    # s_theta_param = pyro.param("scale_ability", torch.ones(num_legs, device=device), constraint=constraints.positive)
    m_b_param = pyro.param("loc_beta", torch.zeros(k_dim, num_votes, device=device))
    s_b_param = pyro.param("scale_beta", torch.empty(k_dim, num_votes, device=device).fill_(1.e2), constraint=constraints.positive)
    m_b_param = pyro.param("loc_beta", torch.zeros(num_votes, device=device))
    s_b_param = pyro.param("scale_beta", torch.empty(k_dim, num_votes, device=device).fill_(1.e2), constraint=constraints.positive)

    # guide distributions
    with pyro.plate('thetas', num_legs, device=device):
        # Fix ideal points to be mean zero and std 1
        dist_theta = dist.Normal(torch.zeros(k_dim, device=device), torch.ones(dim, device=device))
        pyro.sample('theta', dist_theta)
    with pyro.plate('betas', num_votes, device=device):
        dist_b = dist.Normal(m_b_param, s_b_param)
        pyro.sample('beta', dist_b)
    with pyro.plate('alphas', num_votes, device=device):
        dist_b = dist.Normal(m_b_param, s_b_param)
        pyro.sample('alpha', dist_b)


# Setup the optimizer
optim = Adam({'lr': 0.1})

# Define the guide
guide = autoguides.AutoNormal(ideal_point_model)
# guide = autoguides.AutoNormal(ideal_point_model, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
# guide = ideal_point_guide(legs, votes, responses, i)

# Setup the variational inference
svi = SVI(ideal_point_model, guide, optim, loss=Trace_ELBO())
# svi.step(legs, votes, responses, k_dim=k_dim)
# svi.step(torch.tensor([1,1]), torch.tensor([0,1]), torch.tensor([0.,1.]), k_dim=k_dim)

# Run variational inference
pyro.clear_param_store()
for j in range(2000):
    loss = svi.step(legs, votes, responses, k_dim)
    if j % 100 == 0:
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))

# Print parameter values
# for name in pyro.get_param_store().get_all_param_names():
#     print(name)
#     if gpu:
#         val = pyro.param(name).data.cpu().numpy()
#     else:
#         val = pyro.param(name).data.numpy()
#     print(val)

# Get the parameters into pandas dataframes
ideal_points = pd.concat(
                [pd.DataFrame(pyro.param("AutoNormal.locs.theta").data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
                 pd.DataFrame(pyro.param("AutoNormal.scales.theta").data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
                 ], axis=1)
polarity = pd.concat(
                [pd.DataFrame(pyro.param("AutoNormal.locs.beta").data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
                 pd.DataFrame(pyro.param("AutoNormal.scales.beta").data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
                 ], axis=1)
popularity = pd.DataFrame({"loc": pyro.param('AutoNormal.locs.alpha').data.numpy(), "scale": pyro.param('AutoNormal.scales.alpha').data.numpy()})

# Generate predictions for model evaluation
# The iterative process is necessary because we used an autoguide
preds_list = []
for _ in range(1000):
    guide_trace = pyro.poutine.trace(guide).get_trace(legs, votes)
    preds_list.append(pyro.poutine.replay(ideal_point_model, guide_trace)(legs, votes))
preds = torch.stack(preds_list)

# Calculate log loss and accuracy scores, in sample
loss = torch.nn.BCELoss()
loss(preds.mean(axis=0), responses)
accuracy_score(responses, 1 * (preds.mean(axis=0) >= 0.5))

# Generate test data predictions
preds_list = []
for _ in range(1000):
    guide_trace = pyro.poutine.trace(guide).get_trace(legs_test, votes_test)
    preds_list.append(pyro.poutine.replay(ideal_point_model, guide_trace)(legs_test, votes_test))
preds = torch.stack(preds_list)

# Calculate log loss and accuracy scores, out of sample
predictions = pd.Series(preds.mean(axis=0).numpy())
loss = torch.nn.BCELoss()
loss(preds.mean(axis=0), responses_test)
accuracy_score(responses_test, 1 * (preds.mean(axis=0) >= 0.5))

#############
# MCMC Time #
#############

# Initialize the MCMC to our estimates from the variational model
# Speeds things up and requires less burn-in
init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                    "beta": pyro.param("AutoNormal.locs.beta").data,
                                    "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                    })

# Set up sampling alrgorithm
nuts_kernel = NUTS(ideal_point_model, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
# For real inference should probably increase the number of samples, but this is slow and enough to test
hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
# Run the model
hmc_posterior.run(legs, votes, responses, k_dim=k_dim)

# Generate model predictions based on the posterior samples
posterior_predictive = pyro.infer.predictive.Predictive(ideal_point_model, hmc_posterior.get_samples())
preds = posterior_predictive(legs, votes)["obs"].mean(axis=0).flatten(0)
preds_test = posterior_predictive(legs_test, votes_test)["obs"].mean(axis=0).flatten(0)

# In sample metrics
loss = torch.nn.BCELoss()
train_log_loss = loss(preds, responses)
print("Variational log loss: {}".format(train_log_loss))
train_accuracy = accuracy_score(responses, preds.numpy() >= 0.5)
print("Variational accuracy: {}".format(train_accuracy))

# Out of sample metrics
loss = torch.nn.BCELoss()
test_log_loss = loss(preds_test, responses_test)
print("Variational test log loss: {}".format(test_log_loss))
test_accuracy = accuracy_score(responses_test, preds_test.numpy() >= 0.5)
print("Variational test accuracy: {}".format(test_accuracy))

# Gthe MCMC results into a dataframe
samples = hmc_posterior.get_samples()

ideal_points_mcmc = pd.concat([
                        pd.DataFrame(samples["theta"].mean(axis=0).numpy(), columns=['loc_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
                        pd.DataFrame(samples["theta"].std(axis=0).numpy(), columns=['scale_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
                        ], axis=1)

# Compare thre results of the two processes
comp_ideal = pd.concat([ideal_points, ideal_points_mcmc], axis=1)
# comp_ideal.describe()
# comp_ideal.corr()
# comp_ideal.plot(kind='scatter', x="loc_1", y="loc_1_mcmc")
# comp_ideal.plot(kind='scatter', x="scale_1_mcmc", y="scale_1")

import seaborn as sns
sns.distplot(pd.Series(samples["theta"][:, 0, 0].numpy()), bins=25)

temp_d = dist.Normal(ideal_points.loc[0, "loc_1"], ideal_points.loc[0, "scale_1"])
sns.distplot(pd.Series([temp_d().item() for k in range(100)]), bins=25)


def normalize_ideal_points(theta, beta, alpha, verify_predictions=False):
    '''Normalize all ideal points to N(0,1) and adjust other parameters accordingly

    Inputs are assumed to be 3 dimensional tensors where the first dimension is the number
    of monte carlo samples, the second is the length of the parameter, and the third is the
    dimension of the model. If the results are from a variational bayes estimate, simply add
    a singleton first dimension using .unsqueeze(0) prior to passing the arguments.

    Args:
        theta (tensor):
        beta (tensor):
        alpha (tensor):
        verify_predictions (bool): Verify that the transformed parameters yield same outcome,
            up to a certain tolerance. Useful for verification, but can be memory intensive in
            large samples because it calculates predictions for all combinations of votes
            and legislators.
    '''

    ideal_mean_all = theta.mean(axis=(0, 1))
    ideal_std_all = theta.std(axis=(0, 1))

    theta_t = (theta - ideal_mean_all) / ideal_std_all
    theta_t.mean(axis=(0,1))
    beta_t = beta * ideal_std_all
    alpha_t = alpha + (beta * ideal_mean_all).sum(axis=2)

    if verify_predictions:
        prediction = torch.matmul(beta, theta.transpose(1, 2)).add(alpha.unsqueeze(2))
        prediction_t = torch.matmul(beta_t, theta_t.transpose(1, 2)).add(alpha_t.unsqueeze(2))

        prediction - prediction_t

        assert torch.allclose(prediction, prediction_t, atol=1e-05), "Transformation does not match!"

    return {"theta": theta_t, "beta": beta_t, "alpha": alpha_t}


transformed_params_mcmc = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"])
transformed_params = normalize_ideal_points(pyro.param("AutoNormal.locs.theta").data.unsqueeze(0),
                                            pyro.param("AutoNormal.locs.beta").data.unsqueeze(0),
                                            pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0))
