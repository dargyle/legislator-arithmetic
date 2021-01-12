import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides

from pyro.infer.autoguide.initialization import init_to_value

from pyro.ops.stats import hpdi, waic

import pandas as pd
import numpy as np

from data_generation.data_processing import process_data, format_model_data
from data_generation.random_votes import generate_nominate_votes

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


pyro.enable_validation(True)


def bayes_irt_basic(legs, votes, y=None, k_dim=1, device=None):
    """Define a core ideal point model

    Args:
        legs: a tensor of legislator ids
        votes: a tensor of vote ids
        y: a tensor of vote choices
        k_dim: desired dimensions of the models
    """
    # Set some constants
    n_legs = len(set(legs.numpy()))
    n_votes = len(set(votes.numpy()))

    # Set up parameter plates for all of the parameters
    with pyro.plate('thetas', n_legs, dim=-2, device=device):
        ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', n_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', n_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    # Combine parameters
    logit = torch.sum(ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes]

    if y is not None:
        # If training outcomes are provided, run the sampling statement for the given data
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        # If no training outcomes are provided, return the samples from the predicted distributions
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


def bayes_irt_full(legs, votes, y=None, covariates=None, time_passed=None, k_dim=1, device=None):
    """Define a core ideal point model

    Args:
        legs: a tensor of legislator ids
        votes: a tensor of vote ids
        y: a tensor of vote choices
        k_dim: desired dimensions of the models
    """
    n_legs = len(set(legs.numpy()))
    n_votes = len(set(votes.numpy()))
    if covariates is not None:
        n_covar = covariates.shape[1]

    # Set up parameter plates for all of the parameters
    if time_passed is not None:
        k_time = time_passed.shape[1]

        with pyro.plate('thetas', n_legs, dim=-3, device=device):
            ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, k_time, device=device), torch.ones(k_dim, k_time, device=device)))
    else:
        with pyro.plate('thetas', n_legs, dim=-2, device=device):
            ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', n_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', n_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    # Slice the parameter arrays according to the data
    # Allows vectorization of samples later
    if time_passed is not None:
        use_ideal_point = torch.sum(ideal_point[legs] * time_passed[votes].unsqueeze(dim=1), axis=2)
    else:
        use_ideal_point = torch.index_select(ideal_point, -2, legs)
    use_polarity = torch.index_select(polarity, -2, votes)
    use_popularity = torch.index_select(popularity, -1, votes)

    temp_ideal = torch.sum(use_ideal_point * use_polarity, dim=-1)

    if covariates is not None:
        with pyro.plate('coefs', n_covar, device=device):
            coef = pyro.sample('coef', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))
        covar_combo = torch.matmul(covariates, coef.unsqueeze(-1)).squeeze()

        logit = temp_ideal + use_popularity + covar_combo
    else:
        logit = temp_ideal + use_popularity
    if y is not None:
        # If training outcomes are provided, run the sampling statement for the given data
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        # If no training outcomes are provided, return the samples from the predicted distributions
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


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
    Returns:

    '''

    ideal_mean_all = theta.mean(axis=(0, 1))
    ideal_std_all = theta.std(axis=(0, 1))

    theta_t = (theta - ideal_mean_all) / ideal_std_all
    beta_t = beta * ideal_std_all
    alpha_t = alpha + (beta * ideal_mean_all).sum(axis=2)

    if verify_predictions:
        prediction = torch.matmul(beta, theta.transpose(1, 2)).add(alpha.unsqueeze(2))
        prediction_t = torch.matmul(beta_t, theta_t.transpose(1, 2)).add(alpha_t.unsqueeze(2))

        prediction - prediction_t

        assert torch.allclose(prediction, prediction_t, atol=1e-05), "Transformation does not match!"

    results = {"theta": theta_t, "beta": beta_t, "alpha": alpha_t}

    return results


if __name__ == '__main__':
    logger.info("Running some basic model tests on synthetic data")
    # Set up environment
    gpu = False
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info("Generate a test dataset that has 2 dimensions and has a covariate")
    votes = generate_nominate_votes(n_leg=50, n_votes=100, beta=15.0, beta_covar=5.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
    vote_df = votes.reset_index()

    # Process the data for the model
    k_dim = 2
    data_params = dict(
                   vote_df=vote_df,
                   congress_cutoff=110,
                   k_dim=k_dim,
                   k_time=0,
                   covariates_list=["in_majority"],
                   unanimity_check=False,
                   validation_split=0.0,
                   )
    vote_data = process_data(**data_params)
    custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

    x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

    logger.info("Convert training and test data to tensors")
    legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
    votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
    responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
    covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)

    legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
    votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
    responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
    covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)

    logger.info("Test a model with covariates")
    logger.info("Setup the Bayesian Model")
    # Choose the optimizer used by the variational algorithm
    optim = Adam({'lr': 0.1})

    # Define the guide, intialize to the values returned from process data
    guide = autoguides.AutoNormal(bayes_irt_full, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
    # guide = ideal_point_guide(legs, votes, responses, i)

    # Setup the variational inference
    svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())
    # svi.step(legs, votes, responses, covariates, k_dim=k_dim)

    logger.info("Run the variational inference")
    pyro.clear_param_store()
    for j in range(2000):
        loss = svi.step(legs, votes, responses, covariates=covariates, k_dim=k_dim)
        if j % 100 == 0:
            logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))

    logger.info("Now do mcmc sampling")
    # Set initial values to the results from variational Bayes
    # This is not necessary, but lets us use a shorter warmup period because it converges faster
    init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                        "beta": pyro.param("AutoNormal.locs.beta").data,
                                        "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                        "coef": pyro.param("AutoNormal.locs.coef").data,
                                        })

    # Set up sampling alrgorithm
    nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
    # For real inference should probably increase the number of samples, but this is slow and enough to test
    hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
    # Run the model
    hmc_posterior.run(legs, votes, responses, covariates, k_dim=k_dim)

    # Summarize the results
    hmc_posterior.summary()

    # samples = hmc_posterior.get_samples()
    # posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, samples)
    # vectorized_trace = posterior_predictive.get_vectorized_trace(legs, votes, covariates=covariates)

    logger.info('Now run a 2 dimenional model with covariates and a time varying component')
    # Have to reprocess the data to return the time infomation
    k_dim = 2
    # Note that the key difference here is k_time=1
    data_params = dict(
                   vote_df=vote_df,
                   congress_cutoff=110,
                   k_dim=k_dim,
                   k_time=1,
                   covariates_list=["in_majority"],
                   unanimity_check=False,
                   validation_split=0.0,
                   )
    vote_data = process_data(**data_params)
    custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

    x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

    logger.info("Convert training and test data to tensors")
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

    # Choose the optimizer used by the variational algorithm
    optim = Adam({'lr': 0.1})

    # Define the guide
    guide = autoguides.AutoNormal(bayes_irt_full)
    # guide = ideal_point_guide(legs, votes, responses, i)

    # Setup the variational inference
    svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())
    # svi.step(legs, votes, y=responses, covariates=covariates, time_passed=time_passed, k_dim=k_dim)

    logger.info("Run variational inference")
    pyro.clear_param_store()
    for j in range(2000):
        loss = svi.step(legs, votes, y=responses, covariates=covariates, time_passed=time_passed, k_dim=k_dim)
        if j % 100 == 0:
            logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))

    logger.info("Now do mcmc sampling")
    # Set initial values to the results from variational Bayes
    # This is not necessary, but lets us use a shorter warmup period because it converges faster
    init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                        "beta": pyro.param("AutoNormal.locs.beta").data,
                                        "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                        "coef": pyro.param("AutoNormal.locs.coef").data,
                                        })

    # Set up sampling alrgorithm
    nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
    # For real inference should probably increase the number of samples, but this is slow and enough to test
    hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
    # Run the model
    hmc_posterior.run(legs, votes, y=responses, covariates=covariates, time_passed=time_passed, k_dim=k_dim)

    # Summarize the results
    hmc_posterior.summary()
