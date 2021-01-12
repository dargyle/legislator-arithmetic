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
import numpy as np

from data_generation.data_processing import process_data, format_model_data
from data_generation.random_votes import generate_nominate_votes

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

from leg_math.pytorch_bayes import bayes_irt_basic, bayes_irt_full, normalize_ideal_points

pyro.enable_validation(True)

# Set up environment
# Untested with GPU, that's part of what this is about
gpu = False

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

k_dim = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
               k_dim=k_dim,
               k_time=0,
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

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)

# Set some constants
n_legs = len(set(legs.numpy()))
n_votes = len(set(votes.numpy()))

# Setup the optimizer
optim = Adam({'lr': 0.1})

# Define the guide
guide = autoguides.AutoNormal(bayes_irt_full)
# guide = autoguides.AutoNormal(bayes_irt_basic, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
# guide = ideal_point_guide(legs, votes, responses, i)

# Setup the variational inference
svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())
# svi.step(legs, votes, responses, k_dim=k_dim)
# svi.step(torch.tensor([1,1]), torch.tensor([0,1]), torch.tensor([0.,1.]), k_dim=k_dim)

# Run variational inference
pyro.clear_param_store()
for j in range(2000):
    loss = svi.step(legs, votes, y=responses, covariates=covariates, k_dim=k_dim, device=device)
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
    guide_trace = pyro.poutine.trace(guide).get_trace(legs, votes, covariates=covariates)
    preds_list.append(pyro.poutine.replay(bayes_irt_full, guide_trace)(legs, votes, covariates=covariates))
preds = torch.stack(preds_list)

# Calculate log loss and accuracy scores, in sample
loss = torch.nn.BCELoss()
loss(preds.mean(axis=0), responses)
accuracy_score(responses, 1 * (preds.mean(axis=0) >= 0.5))

# Generate test data predictions
preds_list = []
for _ in range(1000):
    guide_trace = pyro.poutine.trace(guide).get_trace(legs_test, votes_test, covariates=covariates_test)
    preds_list.append(pyro.poutine.replay(bayes_irt_full, guide_trace)(legs_test, votes_test, covariates=covariates_test))
preds = torch.stack(preds_list)

# Calculate log loss and accuracy scores, out of sample
predictions = pd.Series(preds.mean(axis=0).numpy())
loss = torch.nn.BCELoss()
loss(preds.mean(axis=0), responses_test)
accuracy_score(responses_test, 1 * (preds.mean(axis=0) >= 0.5))

pyro.param("AutoNormal.locs.coef").data
pyro.param("AutoNormal.scales.coef").data

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
nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
# For real inference should probably increase the number of samples, but this is slow and enough to test
hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
# Run the model
hmc_posterior.run(legs, votes, responses, covariates=covariates, k_dim=k_dim)

# Generate model predictions based on the posterior samples
posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, hmc_posterior.get_samples())
preds = posterior_predictive(legs, votes, covariates=covariates)["obs"].mean(axis=0).flatten(0)
preds_test = posterior_predictive(legs_test, votes_test, covariates=covariates_test)["obs"].mean(axis=0).flatten(0)

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

sns.distplot(pd.DataFrame(samples["coef"].numpy()))

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


from pyro.ops.stats import hpdi, waic
hpdi(samples["coef"], prob=0.95)
waic(hmc_posterior)


# samples = hmc_posterior.get_samples(group_by_chain=False)
# predictive = pyro.infer.Predictive(self.model, samples)
# vectorized_trace = predictive.get_vectorized_trace(*self._args, **self._kwargs)
# for obs_name in self.observations.keys():
#     obs_site = vectorized_trace.nodes[obs_name]
#     log_like = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
#     shape = (self.nchains, self.ndraws) + log_like.shape[1:]
#     data[obs_name] = np.reshape(log_like, shape)

posterior_predictive(legs, votes, covariates=covariates)
vectorized_trace = posterior_predictive.get_vectorized_trace(legs, votes, covariates=covariates)


from pyro.infer.predictive import _predictive
_predictive(self.model, posterior_samples, self.num_samples, return_trace=True, model_args=args, model_kwargs=kwargs)
hmc_posterior.get_samples()["coef"]
torch.mm(covariates, hmc_posterior.get_samples()["coef"])
_predictive(bayes_irt_full, hmc_posterior.get_samples(), 250, return_trace=True, model_args=[legs, votes], model_kwargs={"covariates": covariates})




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



vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "Senate"]

# first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
# first_session.columns = ["first_session", "last_session"]
# first_session["sessions_served"] = first_session["last_session"] - first_session["first_session"]
# # first_session["first_session"].value_counts()
# vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
# vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]

# pd.DataFrame(np.stack([(vote_df["time_passed"] ** i).values for i in range(0, k_time + 1)]).transpose())

k_dim = 2
k_time = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
               k_dim=k_dim,
               k_time=k_time,
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
# covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
time_passed = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)

# pd.DataFrame(np.stack(vote_data["time_passed_train"]).transpose()).sort_values(1)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
# covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# time_tensor = torch.cat([torch.ones(time_passed.shape[0], device=device).unsqueeze(-1), time_passed], axis=1)
time_tensor = time_passed
# time_tensor_test = torch.cat([torch.ones(time_passed_test.shape[0], device=device).unsqueeze(-1), time_passed_test], axis=1)
time_tensor_test = time_passed_test

sessions_served = torch.tensor(vote_data["sessions_served"])

# Set some constants
n_legs = len(set(legs.numpy()))
n_votes = len(set(votes.numpy()))
# n_covar = covariates.shape[1]

wnom_model = wnom_full(n_legs, n_votes, k_dim, pretrained=custom_init_values, k_time=data_params["k_time"])

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)

# w = wnom_model.ideal_points[torch.arange(0, 100)]

losses = []
accuracies = []
test_losses = []
test_accuracies = []
for t in tqdm(range(5000)):
    y_pred = wnom_model(legs, votes, time_tensor, sessions_served)
    loss = criterion(y_pred, responses)

    with torch.no_grad():
        # accuracy = accuracy_score(responses, y_pred.numpy() >= 0.5)
        accuracy = ((y_pred > 0) == responses).sum().item() / len(responses)

        y_pred_test = wnom_model(legs_test, votes_test, time_tensor_test)
        loss_test = criterion(y_pred_test, responses_test)
        # accuracy_test = accuracy_score(responses_test, y_pred_test.numpy() >= 0.5)
        accuracy_test = ((y_pred_test > 0) == responses_test).sum().item() / len(responses_test)

    if t % 100 == 99:
        print(t, loss.item())

    losses.append(loss.item())
    accuracies.append(accuracy)
    test_losses.append(loss_test.item())
    test_accuracies.append(accuracy_test)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)

wnom_model.ideal_points[torch.arange(0, n_legs)]
time_tensor.unsqueeze(-1).shape
wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1)
torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1)
torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1).norm(dim=1)

wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)

pd.DataFrame(wnom_model.ideal_points[torch.arange(0, n_legs)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
pd.DataFrame(wnom_model.ideal_points[torch.arange(0, n_legs), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
wnom_model.ideal_points[torch.arange(0, n_legs)].sum(dim=2).norm(dim=1).max()

wnom_model.beta
wnom_model.w

wnom_model.ideal_points.shape
initial_ideal = wnom_model.ideal_points[torch.arange(0, n_legs), :, 0]
initial_ideal.shape
last_session_ideal = initial_ideal + wnom_model.ideal_points[torch.arange(0, n_legs), :, 1] * sessions_served.unsqueeze(-1) + wnom_model.ideal_points[torch.arange(0, n_legs), :, 2] * (sessions_served ** 2).unsqueeze(-1)

ideal_points = wnom_model.ideal_points[torch.arange(0, n_legs), :, :]
asdf = torch.clone(initial_ideal)
for kk in range(1, ideal_points.shape[2]):
    asdf += ideal_points[:, : , kk] * sessions_served.pow(kk).unsqueeze(-1)
asdf

last_session_ideal.shape
pd.Series(last_session_ideal.norm(dim=1).detach().numpy()).hist()
asdf.norm(dim=1).max()
pd.DataFrame(asdf.detach().numpy()).plot(kind="scatter", x=0, y=1)
