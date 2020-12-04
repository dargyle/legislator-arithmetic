import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

import pyro.contrib.autoguide as autoguide

import pandas as pd

from functools import partial


class MultipleParamLog:
    def __init__(self, priors, device, num_votes, num_legs):
        self.priors = priors
        self.device = device
        self.num_votes = num_votes
        self.num_legs = num_legs
        # self.dim = dim

    def model_vague(self, legs, votes, obs):
        with pyro.plate('thetas', self.num_legs, device=self.device):
            ideal_point = pyro.sample('theta', dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)))

        with pyro.plate('bs', self.num_votes, device=self.device):
            polarity = pyro.sample('b', dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(5.0, device=self.device)))

        with pyro.plate('cs', self.num_votes, device=self.device):
            popularity = pyro.sample('c', dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.e3, device=self.device)))

        with pyro.plate('observe_data', obs.size(0), device=self.device):
            pyro.sample("obs", dist.Bernoulli(logits=ideal_point[legs] * polarity[votes] + popularity[votes]), obs=obs)

    def guide_vague(self, legs, votes, obs):
        # register learnable params in the param store
        m_theta_param = pyro.param("loc_ideal_point", torch.zeros(self.num_legs, device=self.device))
        s_theta_param = pyro.param("scale_ideal_point", torch.ones(self.num_legs, device=self.device),
                                   constraint=constraints.positive)
        m_b_param = pyro.param("loc_polarity", torch.zeros(self.num_votes, device=self.device))
        s_b_param = pyro.param("scale_polarity", torch.empty(self.num_votes, device=self.device).fill_(1.e2),
                               constraint=constraints.positive)
        m_c_param = pyro.param("loc_popularity", torch.zeros(self.num_votes, device=self.device))
        s_c_param = pyro.param("scale_popularity", torch.empty(self.num_votes, device=self.device).fill_(1.e2),
                               constraint=constraints.positive)

        # guide distributions
        with pyro.plate('thetas', self.num_legs, device=self.device):
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample('theta', dist_theta)
        with pyro.plate('bs', self.num_votes, device=self.device):
            dist_b = dist.Normal(m_b_param, s_b_param)
            pyro.sample('b', dist_b)
        with pyro.plate('cs', self.num_votes, device=self.device):
            dist_c = dist.Normal(m_c_param, s_c_param)
            pyro.sample('c', dist_c)

    def model_hierarchical(self, legs, votes, obs):
        mu_b = pyro.sample('mu_b', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e6, device=self.device)))
        u_b = pyro.sample('u_b', dist.Gamma(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))
        mu_theta = pyro.sample('mu_theta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e6, device=self.device)))
        u_theta = pyro.sample('u_theta', dist.Gamma(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))
        with pyro.plate('thetas', self.num_legs, device=self.device):
            ideal_point = pyro.sample('theta', dist.Normal(mu_theta, 1. / u_theta))
        with pyro.plate('bs', self.num_votes, device=self.device):
            polarity = pyro.sample('b', dist.Normal(mu_b, 1. / u_b))
        with pyro.plate('observe_data', obs.size(0)):
            pyro.sample("obs", dist.Bernoulli(logits=ideal_point[legs] - polarity[votes]), obs=obs)

    def guide_hierarchical(self, legs, votes, obs):
        loc_mu_b_param = pyro.param('loc_mu_b', torch.tensor(0., device=self.device))
        scale_mu_b_param = pyro.param('scale_mu_b', torch.tensor(1.e2, device=self.device),
                                      constraint=constraints.positive)
        loc_mu_theta_param = pyro.param('loc_mu_theta', torch.tensor(0., device=self.device))
        scale_mu_theta_param = pyro.param('scale_mu_theta', torch.tensor(1.e2, device=self.device),
                                          constraint=constraints.positive)
        alpha_b_param = pyro.param('alpha_b', torch.tensor(1., device=self.device),
                                   constraint=constraints.positive)
        beta_b_param = pyro.param('beta_b', torch.tensor(1., device=self.device),
                                  constraint=constraints.positive)
        alpha_theta_param = pyro.param('alpha_theta', torch.tensor(1., device=self.device),
                                       constraint=constraints.positive)
        beta_theta_param = pyro.param('beta_theta', torch.tensor(1., device=self.device),
                                      constraint=constraints.positive)
        m_theta_param = pyro.param('loc_ideal_point', torch.zeros(self.num_legs, device=self.device))
        s_theta_param = pyro.param('scale_ideal_point', torch.ones(self.num_legs, device=self.device),
                                   constraint=constraints.positive)
        m_b_param = pyro.param('loc_polarity', torch.zeros(self.num_votes, device=self.device))
        s_b_param = pyro.param('scale_polarity', torch.ones(self.num_votes, device=self.device),
                               constraint=constraints.positive)

        # sample statements
        mu_b = pyro.sample('mu_b', dist.Normal(loc_mu_b_param, scale_mu_b_param))
        u_b = pyro.sample('u_b', dist.Gamma(alpha_b_param, beta_b_param))
        mu_theta = pyro.sample('mu_theta', dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        u_theta = pyro.sample('u_theta', dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate('thetas', self.num_legs, device=self.device):
            pyro.sample('theta', dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate('bs', self.num_votes, device=self.device):
            pyro.sample('b', dist.Normal(m_b_param, s_b_param))

    def fit(self, legs, votes, responses, num_epochs):
        optim = Adam({'lr': 0.1})
        if self.priors == 'vague':
            svi = SVI(self.model_vague, autoguide.AutoNormal(self.model_vague), optim, loss=Trace_ELBO())
        else:
            svi = SVI(self.model_hierarchical, self.guide_hierarchical, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(legs, votes, responses)
            if j % 100 == 0:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        values = ['loc_polarity', 'scale_polarity', 'loc_ideal_point', 'scale_ideal_point']

    def fit_MCMC(self, legs, votes, responses, num_epochs):
        sites = ['theta', 'b']
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(legs, votes, responses)
        theta_sum = self.summary(hmc_posterior, ['theta']).votes()
        b_sum = self.summary(hmc_posterior, ['b']).votes()
        print(theta_sum)
        print(b_sum)

    def summary(self, traces, sites):
        marginal = EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats




def model_vague(legs, votes, obs, dim):
    with pyro.plate('thetas', num_legs, device=device):
        ideal_point = pyro.sample('theta', dist.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)))
        pyro.sample('theta', dist.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)))

    with pyro.plate('bs', num_votes, device=device):
        polarity = pyro.sample('b', dist.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)))

    with pyro.plate('cs', num_votes, device=device):
        popularity = pyro.sample('c', dist.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device)))

    with pyro.plate('observe_data', obs.size(0), device=device):
        pyro.sample("obs", dist.Bernoulli(logits=torch.dot(ideal_point[legs], polarity[votes]) + popularity[votes]), obs=obs)

def guide_vague(legs, votes, obs, dim):
    m_theta_param = pyro.param("loc_ideal_point", torch.zeros(num_legs, dim, device=device))
    s_theta_param = pyro.param("scale_ideal_point", torch.ones(num_legs, device=device),
                               constraint=constraints.positive_definite)
    m_b_param = pyro.param("loc_polarity", torch.zeros(num_votes, dim, device=device))
    s_b_param = pyro.param("scale_polarity", torch.empty(num_votes, device=device).fill_(1.e2),
                           constraint=constraints.positive_definite)
    m_c_param = pyro.param("loc_popularity", torch.zeros(num_votes, device=device))
    s_c_param = pyro.param("scale_popularity", torch.empty(num_votes, device=device).fill_(1.e2),
                           constraint=constraints.positive_definite)

    # guide distributions
    with pyro.plate('thetas', num_legs, device=device):
        dist_theta = dist.MultivariateNormal(m_theta_param, s_theta_param)
        pyro.sample('theta', dist_theta)
    with pyro.plate('bs', num_votes, device=device):
        dist_b = dist.MultivariateNormal(m_b_param, s_b_param)
        pyro.sample('b', dist_b)
    with pyro.plate('cs', num_votes, device=device):
        dist_c = dist.MultivariateNormal(m_c_param, s_c_param)
        pyro.sample('c', dist_c)


def model(data):
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)

def guide(legs, votes, obs, dim):
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([T]))
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("mu_plate", T):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))



class MultipleDimParamLog:
    def __init__(self, priors, device, num_votes, num_legs, dim):
        self.priors = priors
        self.device = device
        self.num_votes = num_votes
        self.num_legs = num_legs
        self.dim = dim

    def model_vague(self, legs, votes, obs):
        with pyro.plate('thetas', self.num_legs, device=self.device):
            ideal_point = pyro.sample('theta', dist.MultivariateNormal(torch.zeros(self.dim, device=self.device), torch.eye(self.dim, device=self.device)))
            # pyro.sample('theta', dist.MultivariateNormal(torch.zeros(self.dim, device=device), torch.eye(self.dim, device=device)))

        with pyro.plate('bs', self.num_votes, device=self.device):
            polarity = pyro.sample('b', dist.MultivariateNormal(torch.zeros(self.dim, device=self.device), torch.eye(self.dim, device=self.device)))

        with pyro.plate('cs', self.num_votes, device=self.device):
            popularity = pyro.sample('c', dist.MultivariateNormal(torch.zeros(self.dim, device=self.device), torch.eye(self.dim, device=self.device)))

        with pyro.plate('observe_data', obs.size(0), device=self.device):
            pyro.sample("obs", dist.Bernoulli(logits=torch.dot(ideal_point[legs], polarity[votes]) + popularity[votes]), obs=obs)

    # def guide_vague(self, legs, votes, obs):
    #     # register learnable params in the param store
    #     m_theta_param = pyro.param("loc_ideal_point", torch.zeros(self.num_legs, self.dim, device=self.device))
    #     s_theta_param = pyro.param("scale_ideal_point", torch.ones(self.num_legs, device=self.device),
    #                                constraint=constraints.positive_definite)
    #     m_b_param = pyro.param("loc_polarity", torch.zeros(self.num_votes, self.dim, device=self.device))
    #     s_b_param = pyro.param("scale_polarity", torch.empty(self.num_votes, device=self.device).fill_(1.e2),
    #                            constraint=constraints.positive_definite)
    #     m_c_param = pyro.param("loc_popularity", torch.zeros(self.num_votes, device=self.device))
    #     s_c_param = pyro.param("scale_popularity", torch.empty(self.num_votes, device=self.device).fill_(1.e2),
    #                            constraint=constraints.positive_definite)
    #
    #     # guide distributions
    #     with pyro.plate('thetas', self.num_legs, device=self.device):
    #         dist_theta = dist.MultivariateNormal(m_theta_param, s_theta_param)
    #         pyro.sample('theta', dist_theta)
    #     with pyro.plate('bs', self.num_votes, device=self.device):
    #         dist_b = dist.MultivariateNormal(m_b_param, s_b_param)
    #         pyro.sample('b', dist_b)
    #     with pyro.plate('cs', self.num_votes, device=self.device):
    #         dist_c = dist.MultivariateNormal(m_c_param, s_c_param)
    #         pyro.sample('c', dist_c)

    def fit(self, legs, votes, responses, num_epochs):
        optim = Adam({'lr': 0.1})
        if self.priors == 'vague':
            svi = SVI(self.model_vague, autoguide.AutoMultivariateNormal(self.model_vague), optim, loss=Trace_ELBO())
        else:
            svi = SVI(self.model_hierarchical, self.guide_hierarchical, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(legs, votes, responses)
            if j % 100 == 0:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        values = ['loc_polarity', 'scale_polarity', 'loc_ideal_point', 'scale_ideal_point']

    def fit_MCMC(self, legs, votes, responses, num_epochs):
        sites = ['theta', 'b']
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(legs, votes, responses)
        theta_sum = self.summary(hmc_posterior, ['theta']).votes()
        b_sum = self.summary(hmc_posterior, ['b']).votes()
        print(theta_sum)
        print(b_sum)

    def summary(self, traces, sites):
        marginal = EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats


import pandas as pd
from data_generation.data_processing import process_data, format_model_data

from constants import DATA_PATH

# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--num-epochs', default=1000, type=int)
# parser.add_argument('--gpu', action='store_true')
# parser.add_argument('--priors', help='[vague, hierarchical]', default='hierarchical')
# parser.add_argument('--model', help='[1PL,2PL]', default='1PL')  # 2pl not implemented yet
# parser.add_argument('--response-patterns', help='file with response pattern data', required=True)
# args = parser.parse_args()



gpu = False

device = torch.device('cpu')
if gpu:
    device = torch.device('cuda')

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

legs = torch.tensor(legs, dtype=torch.long, device=device)
votes = torch.tensor(votes, dtype=torch.long, device=device)
responses = torch.tensor(responses, dtype=torch.float, device=device)

# 3. define model and guide accordingly
# priors = 'hierarchical'
priors = 'vague'
num_epochs = 2500
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

ideal_points_mcmc = pd.DataFrame({"loc_mcmc": samples["theta"].median(axis=0).values, "scale_mcmc": samples["theta"].std(axis=0)})

pd.concat([ideal_points, ideal_points_mcmc], axis=1).corr()
pd.concat([ideal_points, ideal_points_mcmc], axis=1).plot(kind="scatter", x="loc", y="loc_mcmc")
pd.concat([ideal_points, ideal_points_mcmc], axis=1).plot(kind="scatter", x="scale", y="scale_mcmc")

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
