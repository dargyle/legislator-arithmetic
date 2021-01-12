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
        final_ideal_point = torch.sum(ideal_point[legs] * time_passed[votes].unsqueeze(dim=1), axis=2)
        # ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1)
        # ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1)
        # torch.sum(ideal_point[[1, 1]] * time_tensor[[1, 3]].unsqueeze(-1), axis=1)
    else:
        with pyro.plate('thetas', n_legs, dim=-2, device=device):
            final_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', n_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', n_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    # Slice the parameter arrays according to the data
    use_ideal_point = torch.index_select(final_ideal_point, -2, legs)
    use_polarity = torch.index_select(polarity, -2, votes)
    use_popularity = torch.index_select(popularity, -1, votes)

    temp_ideal = torch.sum(use_ideal_point * use_polarity, dim=-1)

    if covariates is not None:
        with pyro.plate('coefs', n_covar, device=device):
            coef = pyro.sample('coef', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))
        # print(covariates.shape)
        # print(coef.shape)
        # print(coef.unsqueeze(-1).shape)
        # print('ideal_shape: {}'.format(final_ideal_point[legs].shape))
        # print('polarity_shape: {}'.format(polarity[votes].shape))
        covar_combo = torch.matmul(covariates, coef.unsqueeze(-1)).squeeze()
        # print('temp_ideal_shape: {}'.format(temp_ideal.shape))
        # print('popularity_shape: {}'.format(popularity[votes].shape))
        # print('covar_combo_shape: {}'.format(covar_combo.shape))
        logit = temp_ideal + use_popularity + covar_combo
    else:
        logit = temp_ideal + use_popularity
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
    beta_t = beta * ideal_std_all
    alpha_t = alpha + (beta * ideal_mean_all).sum(axis=2)

    if verify_predictions:
        prediction = torch.matmul(beta, theta.transpose(1, 2)).add(alpha.unsqueeze(2))
        prediction_t = torch.matmul(beta_t, theta_t.transpose(1, 2)).add(alpha_t.unsqueeze(2))

        prediction - prediction_t

        assert torch.allclose(prediction, prediction_t, atol=1e-05), "Transformation does not match!"

    return {"theta": theta_t, "beta": beta_t, "alpha": alpha_t}


if __name__ == '__main__':
    # Set up environment
    # Untested with GPU, that's part of what this is about
    gpu = False

    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Generate datset that has 2 dimensions and has a covariate
    votes = generate_nominate_votes(n_leg=50, n_votes=100, beta=15.0, beta_covar=5.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
    vote_df = votes.reset_index()

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

    # Convert training and test data to tensors
    legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
    votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
    responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
    covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)

    legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
    votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
    responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
    covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)

    optim = Adam({'lr': 0.1})

    # Define the guide
    # guide = autoguides.AutoNormal(bayes_irt_full)
    guide = autoguides.AutoNormal(bayes_irt_full, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
    # guide = ideal_point_guide(legs, votes, responses, i)

    # Setup the variational inference
    svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())
    # svi.step(legs, votes, responses, covariates, k_dim=k_dim)

    # Run variational inference
    pyro.clear_param_store()
    for j in range(2000):
        loss = svi.step(legs, votes, responses, covariates=covariates, k_dim=k_dim)
        if j % 100 == 0:
            print("[epoch %04d] loss: %.4f" % (j + 1, loss))

    # pyro.param("AutoNormal.locs.theta").data.numpy()
    # pyro.param("AutoNormal.locs.coef").data.numpy()
    # pyro.param("AutoNormal.scales.coef").data.numpy()

    init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                        "beta": pyro.param("AutoNormal.locs.beta").data,
                                        "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                        })

    # Set up sampling alrgorithm
    nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
    # For real inference should probably increase the number of samples, but this is slow and enough to test
    hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500)
    # Run the model
    hmc_posterior.run(legs, votes, responses, covariates, k_dim=k_dim)

    samples_1 = hmc_posterior.get_samples()
    posterior_predictive_1 = pyro.infer.predictive.Predictive(bayes_irt_full, samples_1)
    vectorized_trace_1 = posterior_predictive_1.get_vectorized_trace(legs, votes, covariates=covariates)

    def make_log_joint(model):
        def _log_joint(cond_data, *args, **kwargs):
            conditioned_model = poutine.condition(model, data=cond_data)
            trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)
            return trace.log_prob_sum()
        return _log_joint

    guide_tr = poutine.trace(guide).get_trace(legs, votes, responses, covariates, k_dim=k_dim)
    model_tr = poutine.trace(poutine.replay(bayes_irt_full, trace=guide_tr)).get_trace(legs, votes, responses, covariates, k_dim=k_dim)
    monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()


    poutine.trace(bayes_irt_full, legs, votes, covariates=covariates)


    obs_name = "obs"
    obs_site_1 = vectorized_trace_1.nodes[obs_name]
    log_like_1 = obs_site_1["fn"].log_prob(obs_site_1["value"])

    hmc_posterior.summary()

    waic_1 = waic(log_like_1)
    waic_1
    hpdi_1 = hpdi(samples_1["coef"], prob=0.95)

    pd.Series(samples_1["coef"].numpy()[:, 0]).hist()
    pd.Series(samples_1["theta"].numpy()[:, 1, 0]).plot()

    import arviz as az
    az.waic(hmc_posterior)
    az.loo(hmc_posterior)




    # Define the guide
    # guide = autoguides.AutoNormal(bayes_irt_full)
    guide_2 = autoguides.AutoNormal(bayes_irt_full, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
    # guide = ideal_point_guide(legs, votes, responses, i)

    # Setup the variational inference
    svi_2 = SVI(bayes_irt_full, guide_2, optim, loss=Trace_ELBO())
    # svi.step(legs, votes, responses, covariates, k_dim=k_dim)

    # Run variational inference
    pyro.clear_param_store()
    for j in range(2000):
        loss = svi_2.step(legs, votes, responses, k_dim=k_dim)
        if j % 100 == 0:
            print("[epoch %04d] loss: %.4f" % (j + 1, loss))

    pyro.param("AutoNormal.locs.theta").data.numpy()

    init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                        "beta": pyro.param("AutoNormal.locs.beta").data,
                                        "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                        })

    # Set up sampling alrgorithm
    nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
    # For real inference should probably increase the number of samples, but this is slow and enough to test
    hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
    # Run the model
    hmc_posterior.run(legs, votes, responses, k_dim=k_dim)

    hmc_posterior.summary()

    samples = hmc_posterior.get_samples()
    posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, samples)
    vectorized_trace = posterior_predictive.get_vectorized_trace(legs, votes)
    obs_name = "obs"
    obs_site = vectorized_trace.nodes[obs_name]
    log_like = obs_site["fn"].log_prob(obs_site["value"])

    waic_2 = waic(log_like)

    waic_1
    waic_2

    torch.exp((waic_1[0] - waic_2[0]) / 2)

    import arviz as az
    az.waic(hmc_posterior)
    az.loo(hmc_posterior)





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
    # n_legs = len(set(legs.numpy()))
    # n_votes = len(set(votes.numpy()))

    # Setup the optimizer
    optim = Adam({'lr': 0.1})

    # Define the guide
    guide = autoguides.AutoNormal(bayes_irt_basic)
    # guide = autoguides.AutoNormal(bayes_irt_basic, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
    # guide = ideal_point_guide(legs, votes, responses, i)

    # Setup the variational inference
    svi = SVI(bayes_irt_basic, guide, optim, loss=Trace_ELBO())
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
        preds_list.append(pyro.poutine.replay(bayes_irt_basic, guide_trace)(legs, votes))
    preds = torch.stack(preds_list)

    # Calculate log loss and accuracy scores, in sample
    loss = torch.nn.BCELoss()
    loss(preds.mean(axis=0), responses)
    accuracy_score(responses, 1 * (preds.mean(axis=0) >= 0.5))

    # Generate test data predictions
    preds_list = []
    for _ in range(1000):
        guide_trace = pyro.poutine.trace(guide).get_trace(legs_test, votes_test)
        preds_list.append(pyro.poutine.replay(bayes_irt_basic, guide_trace)(legs_test, votes_test))
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
    nuts_kernel = NUTS(bayes_irt_basic, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
    # For real inference should probably increase the number of samples, but this is slow and enough to test
    hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
    # Run the model
    hmc_posterior.run(legs, votes, responses, k_dim=k_dim)

    # Generate model predictions based on the posterior samples
    posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_basic, hmc_posterior.get_samples())
    preds = posterior_predictive(legs, votes)["obs"].mean(axis=0).flatten(0)
    preds_test = posterior_predictive(legs_test, votes_test)["obs"].mean(axis=0).flatten(0)

    posterior_predictive.get_vectorized_trace(legs, votes)

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


    transformed_params_mcmc = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"])
    transformed_params_mcmc = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"], verify_predictions=True)
    transformed_params = normalize_ideal_points(pyro.param("AutoNormal.locs.theta").data.unsqueeze(0),
                                                pyro.param("AutoNormal.locs.beta").data.unsqueeze(0),
                                                pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0))










    samples = hmc_posterior.get_samples()
    samples.keys()
    pd.DataFrame(samples["coef"].flatten().numpy()).hist()

    samples["coef"][0].shape
    posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, hmc_posterior.get_samples())
    asdf = posterior_predictive.get_vectorized_trace(legs, votes, responses, covariates=covariates)
    asdf.information_criterion()

    num_samples = len(samples["coef"])
    model = bayes_irt_full
    model_args = [legs, votes]
    model_kwargs = {"covariates": covariates}

    import pyro.poutine as poutine
    from pyro.poutine.util import prune_subsample_sites

    def _guess_max_plate_nesting(model, args, kwargs):
        """
        Guesses max_plate_nesting by running the model once
        without enumeration. This optimistically assumes static model
        structure.
        """
        with poutine.block():
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)
        sites = [site for site in model_trace.nodes.values()
                 if site["type"] == "sample"]

        dims = [frame.dim
                for site in sites
                for frame in site["cond_indep_stack"]
                if frame.vectorized]
        max_plate_nesting = -min(dims) if dims else 0
        return max_plate_nesting

    max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)
    vectorize = pyro.plate("_num_predictive_samples", num_samples, dim=-max_plate_nesting-1)
    model_trace = prune_subsample_sites(poutine.trace(model).get_trace(*model_args, **model_kwargs))
    reshaped_samples = {}

    for name, sample in samples.items():
        sample_shape = sample.shape[1:]
        sample = sample.reshape((num_samples,) + (1,) * (max_plate_nesting - len(sample_shape)) + sample_shape)
        reshaped_samples[name] = sample

    reshaped_samples["coef"].shape

    if return_trace:
        trace = poutine.trace(poutine.condition(vectorize(model), reshaped_samples)).get_trace(*model_args, **model_kwargs)
        return trace

    trace.__dict__

    model_trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
    model_trace.__dict__

    from pyro.infer import TracePosterior
    TracePosterior()(bayes_irt_full, hmc_posterior.get_samples())

    hmc_posterior.__dict__.keys()
    posterior_predictive.__dict__.keys()
    trace = poutine.trace(bayes_irt_full).get_trace(*model_args, **model_kwargs)
    trace.log_prob_sum()
    trace.__dict__
    trace.observation_nodes
    trace.compute_log_prob()

    obs_node = None
    log_likelihoods = []
    trace
    for t in trace:
        obs_nodes = trace.observation_nodes
        if len(obs_nodes) > 1:
            raise ValueError("Infomation criterion calculation only works for models "
                             "with one observation node.")
        if obs_node is None:
            obs_node = obs_nodes[0]
        elif obs_node != obs_nodes[0]:
            raise ValueError("Observation node has been changed, expected {} but got {}"
                             .format(obs_node, obs_nodes[0]))

        trace.nodes.keys()
        trace.nodes["theta"].log_prob
        log_likelihoods.append(trace.nodes[obs_node]["fn"].log_prob(trace.nodes[obs_node]["value"]))


    from pyro.ops.stats import hpdi, waic
    waic(trace.log_prob_sum())

    from arviz.data.io_pyro import PyroConverter, from_pyro
    zxcv = PyroConverter(posterior=hmc_posterior, log_likelihood=True)
    zxcv.log_likelihood_to_xarray()

    import arviz as az
    az.waic(zxcv)
    az.waic(hmc_posterior, var_name="coef")

    qwer = from_pyro(posterior=hmc_posterior)
    qwer["log_likelihood"]

    posterior_predictive.get_vectorized_trace(legs, votes, covariates=covariates)


    samples = zxcv.posterior.get_samples(group_by_chain=False)
    predictive = pyro.infer.Predictive(zxcv.model, samples)
    vectorized_trace = predictive.get_vectorized_trace(*zxcv._args, **zxcv._kwargs)
    for obs_name in zxcv.observations.keys():
        obs_site = vectorized_trace.nodes[obs_name]
        log_like = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
        shape = (zxcv.nchains, zxcv.ndraws) + log_like.shape[1:]
        np.reshape(log_like, shape)

    log_like = obs_site["fn"].log_prob(obs_site["value"])
    waic(log_like)
    waic(log_like[0, : , :])

    vectorized_trace.__dict__

    covariates.shape
