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

pyro.enable_validation(True)


# Set up environment
gpu = False

device = torch.device('cpu')
if gpu:
    device = torch.device('cuda')

vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "House"]

# Load data
k_dim = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=116,
               k_dim=k_dim,
               k_time=1,
               covariates_list=[],
               unanimity_check=False,
               )
vote_data = process_data(**data_params)
custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

legs = x_train[0].flatten()
votes = x_train[1].flatten()
responses = vote_data["y_train"].flatten()

num_legs = len(set(legs))
num_votes = len(set(votes))
print(num_votes, num_legs)

legs = torch.tensor(legs, dtype=torch.long, device=device)
votes = torch.tensor(votes, dtype=torch.long, device=device)
responses = torch.tensor(responses, dtype=torch.float, device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)


def ideal_point_model(legs, votes, y=None, k_dim=1):
    with pyro.plate('thetas', num_legs, dim=-2, device=device):
        ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('betas', num_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', num_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    # print('ideal_shape: {}'.format(ideal_point[legs].shape))
    # print('polarity_shape: {}'.format(polarity[votes].shape))
    # print('popularity_shape: {}'.format(popularity[votes].shape))

    logit = torch.sum(ideal_point[legs] * polarity[votes], dim=-1) + popularity[votes]
    # print(logit)
    # print(logit.shape)

    if y is not None:
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


def ideal_point_guide(legs, votes, obs, k_dim=1):
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


optim = Adam({'lr': 0.1})
guide = autoguides.AutoNormal(ideal_point_model, init_loc_fn=init_to_value(values={'theta': custom_init_values}))
# guide = ideal_point_guide(legs, votes, responses, i)
svi = SVI(ideal_point_model, guide, optim, loss=Trace_ELBO())
# svi.step(legs, votes, responses, dim=2)

# legs.shape
# votes.shape
# responses.shape

pyro.clear_param_store()
for j in range(2000):
    loss = svi.step(legs, votes, responses, k_dim)
    if j % 100 == 0:
        print("[epoch %04d] loss: %.4f" % (j + 1, loss))


for name in pyro.get_param_store().get_all_param_names():
    print(name)
    if gpu:
        val = pyro.param(name).data.cpu().numpy()
    else:
        val = pyro.param(name).data.numpy()
    print(val)

pyro.param("AutoNormal.locs.theta").data.numpy().shape
ideal_points = pd.concat(
                [pd.DataFrame(pyro.param("AutoNormal.locs.theta").data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
                 pd.DataFrame(pyro.param("AutoNormal.scales.theta").data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
                 ], axis=1)
polarity = pd.concat(
                [pd.DataFrame(pyro.param("AutoNormal.locs.beta").data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
                 pd.DataFrame(pyro.param("AutoNormal.scales.beta").data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
                 ], axis=1)
popularity = pd.DataFrame({"loc": pyro.param('AutoNormal.locs.alpha').data.numpy(), "scale": pyro.param('AutoNormal.scales.alpha').data.numpy()})

preds = []
for _ in range(1000):
    guide_trace = pyro.poutine.trace(guide).get_trace(legs, votes)
    preds.append(pyro.poutine.replay(ideal_point_model, guide_trace)(legs, votes))
pd.Series(torch.stack(preds).mean(axis=0).numpy()).hist()

pred = pyro.infer.predictive.Predictive(ideal_point_model, num_samples=1000)
pd.Series(pred(legs_test, votes_test)["obs"].mean(axis=0).numpy()[0]).hist()

pred = pyro.infer.predictive.Predictive(ideal_point_model, guide=guide, num_samples=100, return_sites=("obs", "_RETURN"))
pd.DataFrame(pred(legs, votes, responses)["theta"].mean(axis=0).numpy()).hist()
pd.Series(pred(legs_test, votes_test, responses_test)["obs"].mean(axis=0).numpy()[0]).hist()

init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                     "beta": pyro.param("AutoNormal.locs.beta").data,
                                     "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                     })
nuts_kernel = NUTS(ideal_point_model, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
hmc_posterior.run(legs, votes, responses, k_dim=k_dim)

posterior_predictive = pyro.infer.predictive.Predictive(ideal_point_model, hmc_posterior.get_samples()).forward(legs_test, votes_test, obs=None)
pd.DataFrame(posterior_predictive["obs"].mean(axis=0).numpy().transpose()).hist()
asdf = pd.DataFrame(posterior_predictive["obs"].mean(axis=0).numpy().transpose())
asdf["true"] = responses_test.numpy()

posterior_predictive = pyro.infer.predictive.Predictive(model=ideal_point_model, posterior_samples=hmc_posterior.get_samples()).forward()
pyro.infer.predictive.Predictive(ideal_point_model, num_samples=100)(legs_test, votes_test)

hmc_posterior.summary()

samples = hmc_posterior.get_samples()
samples["theta"].shape

pd.Series(samples["theta"][:, 0, 0].numpy()).plot()

ideal_points_mcmc = pd.concat([
                        pd.DataFrame(samples["theta"].mean(axis=0).numpy(), columns=['loc_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
                        pd.DataFrame(samples["theta"].std(axis=0).numpy(), columns=['scale_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
                        ], axis=1)

comp_ideal = pd.concat([ideal_points, ideal_points_mcmc], axis=1)
comp_ideal.describe()
comp_ideal.corr()
comp_ideal.plot(kind='scatter', x="loc_1", y="loc_1_mcmc")
comp_ideal.plot(kind='scatter', x="scale_1_mcmc", y="scale_1")

import seaborn as sns
sns.distplot(pd.Series(samples["theta"][:, 0, 0].numpy()), bins=25)

temp_d = dist.Normal(ideal_points.loc[0, "loc_1"], ideal_points.loc[0, "scale_1"])
sns.distplot(pd.Series([temp_d().item() for k in range(100)]), bins=25)

samples["theta"].shape
samples["beta"].shape
samples["alpha"].shape
samples["alpha"].mean(axis=1)

samples["beta"].transpose(1, 2).shape

# torch.tensordot(samples["theta"], samples["b"].transpose(1, 2), dims=[[1, 2], [2, 1]]).shape

ideal_mean = samples["theta"].mean(axis=0).mean()
ideal_std = samples["theta"].mean(axis=0).std()


theta = samples["theta"].mean(axis=0)
beta = samples["beta"].mean(axis=0)
alpha = samples["alpha"].mean(axis=0)

predictions = torch.mm(beta, theta.transpose(0, 1)) + alpha.unsqueeze(1)
predictions

ideal_point_t = (theta - ideal_mean) / ideal_std
polarity_t = beta * ideal_std
popularity_t = alpha + (beta * ideal_mean).sum(axis=1)

new_predictions = torch.mm(polarity_t, ideal_point_t.transpose(0, 1)) + popularity_t.unsqueeze(1)
new_predictions

torch.allclose(predictions, new_predictions, rtol=1e-03, atol=1e-06)

samples["alpha"].shape
torch.matmul(samples["theta"], samples["beta"].transpose(1, 2)).shape
torch.matmul(samples["theta"], samples["beta"].transpose(1, 2)).add(samples["alpha"].unsqueeze(1)).shape
torch.matmul(samples["beta"], samples["theta"].transpose(1, 2)).transpose(0, 2)


def normalize_ideal_points(theta, beta, alpha, verify_predictions=True):
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


transformed_params = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"])
normalize_ideal_points(pyro.param("AutoNormal.locs.theta").data.unsqueeze(0), pyro.param("AutoNormal.locs.beta").data.unsqueeze(0), pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0))

theta = samples["theta"]
theta - theta.mean(axis=(0,1))
beta = samples["beta"]
alpha = samples["alpha"]

theta = pyro.param("AutoNormal.locs.theta").data.unsqueeze(0)
beta = pyro.param("AutoNormal.locs.beta").data.unsqueeze(0)
alpha = pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0)

transformed_params["theta"].mean(axis=(0, 1))
transformed_params["theta"].std(axis=(0, 1))
pd.DataFrame(transformed_params["theta"].mean(axis=0).numpy()).hist(bins=20)
pd.DataFrame(transformed_params["theta"].mean(axis=0).numpy()).plot(kind="scatter", x=0, y=1)
pd.DataFrame(transformed_params["theta"].mean(axis=0).numpy()).corr()

temp = normalize_ideal_points(pyro.param("AutoNormal.locs.theta").data.unsqueeze(0), pyro.param("AutoNormal.locs.beta").data.unsqueeze(0), pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0))
pd.DataFrame(temp["theta"].mean(axis=0).numpy()).corr()

posterior_predictive = Predictive(...)(**data_dict)

# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--num-epochs', default=1000, type=int)
# parser.add_argument('--gpu', action='store_true')
# parser.add_argument('--priors', help='[vague, hierarchical]', default='hierarchical')
# parser.add_argument('--model', help='[1PL,2PL]', default='1PL')  # 2pl not implemented yet
# parser.add_argument('--response-patterns', help='file with response pattern data', required=True)
# args = parser.parse_args()


# 3. define model and guide accordingly
# priors = 'hierarchical'
priors = 'vague'
num_epochs = 3500
m1 = MultipleParamLog(priors, device, num_votes, num_legs)
m1 = MultipleDimParamLog(priors, device, num_votes, num_legs, 2)

# 4. fit irt leg with svi, trace-elbo loss
m1.fit(legs, votes, responses, num_epochs)

for name in pyro.get_param_store().get_all_param_names():
    print(name)
    if gpu:
        val = pyro.param(name).data.cpu().numpy()
    else:
        val = pyro.param(name).data.numpy()
    print(val)


# ideal_points = pd.DataFrame({"loc": pyro.param("loc_ideal_point").data.numpy(), "scale": pyro.param("scale_ideal_point").data.numpy()})
# polarity = pd.DataFrame({"loc": pyro.param("loc_polarity").data.numpy(), "scale": pyro.param("scale_polarity").data.numpy()})
# popularity = pd.DataFrame({"loc": pyro.param("loc_polarity").data.numpy(), "scale": pyro.param("scale_popularity").data.numpy()})
ideal_points = pd.DataFrame({"loc": pyro.param("AutoNormal.locs.theta").data.numpy(), "scale": pyro.param("AutoNormal.scales.theta").data.numpy()})
polarity = pd.DataFrame({"loc": pyro.param('AutoNormal.locs.b').data.numpy(), "scale": pyro.param('AutoNormal.scales.b').data.numpy()})
popularity = pd.DataFrame({"loc": pyro.param('AutoNormal.locs.c').data.numpy(), "scale": pyro.param('AutoNormal.scales.c').data.numpy()})

from pyro.infer.mcmc import MCMC, NUTS

pyro.get_param_store().clear()
m2 = MultipleParamLog(priors, device, num_votes, num_legs)
nuts_kernel = NUTS(m2.model_vague, adapt_step_size=True)
hmc_posterior = MCMC(nuts_kernel, num_samples=100, warmup_steps=100)
hmc_posterior.run(legs, votes, responses)

hmc_posterior.summary()

samples = hmc_posterior.get_samples()
samples["theta"].mean(axis=0)
samples["theta"].std(axis=0)

ideal_points_mcmc = pd.DataFrame(samples["theta"].std(axis=0).numpy())

pd.concat([ideal_points, ideal_points_mcmc], axis=1).corr()
pd.concat([ideal_points, ideal_points_mcmc], axis=1).plot(kind="scatter", x="scale_1", y="0")
pd.concat([ideal_points, ideal_points_mcmc], axis=1).plot(kind="scatter", x="scale", y="scale_mcmc")

from pyro.infer.mcmc.util import initialize_model, predictive
predictive(m2, samples, legs, votes, responses)

pd.DataFrame(samples["theta"].data.numpy())

samples = {k: v.reshape((-1,) + v.shape[2:]).data.numpy() for k, v in samples.items()}
pd.DataFrame(samples["theta"])

samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
batch_dim = 0
sample_tensor = list(samples.values())[0]
batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
idxs = torch.randint(0, batch_size, size=(num_samples,), device=device)
samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}


hmc_posterior.summary()

theta_sum = m2.summary(hmc_posterior, ['theta']).items()
b_sum = m.summary(hmc_posterior, ['b']).items()
