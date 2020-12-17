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
random_votes = generate_nominate_votes(n_leg=25, n_votes=250, beta=15.0, beta_covar=0.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
vote_df = random_votes.reset_index()

k_dim = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
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
# covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
# time_passed = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
# covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
# time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# Set some constants
num_legs = len(set(legs.numpy()))
num_votes = len(set(votes.numpy()))
# num_covar = covariates.shape[1]


class wnom(nn.Module):
    def __init__(self, num_legs, num_votes, k_dim, pretrained):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom, self).__init__()
        self.ideal_points = nn.Embedding.from_pretrained(pretrained, max_norm=1.0)
        self.yes_points = nn.Embedding(num_embeddings=num_votes, embedding_dim=k_dim)
        nn.init.uniform_(self.yes_points.weight)
        self.no_points = nn.Embedding(num_embeddings=num_votes, embedding_dim=k_dim)
        nn.init.uniform_(self.no_points.weight)

        self.w = nn.Parameter(5 * torch.ones((k_dim)))
        # self.sig = nn.Sigmoid()

    def forward(self, legs, votes):
        """
        Take in the legislator and vote ids and generate a prediction
        """
        distances1 = torch.sum(torch.square(self.ideal_points(legs) - self.yes_points(votes)) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(self.ideal_points(legs) - self.no_points(votes)) * torch.square(self.w), axis=1)

        result = torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2)
        # result = self.sig(torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


wnom_model = wnom(num_legs, num_votes, k_dim, custom_init_values)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(wnom_model.parameters(), amsgrad=True)

losses = []
for t in range(25000):
    y_pred = wnom_model(legs, votes)
    loss = criterion(y_pred, responses)
    if t % 100 == 99:
        print(t, loss.item())

    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)

losses
pd.Series(losses).plot()

true_ideal = random_votes[["coord1D", "coord2D"]]
true_ideal.index = true_ideal.index.droplevel("vote_id")
true_ideal = true_ideal.drop_duplicates()
vote_data.keys()
leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}
true_ideal[["wnom1", "wnom2"]] = wnom_model.ideal_points(torch.tensor(true_ideal.index.map(leg_crosswalk_rev).values)).detach().numpy()

true_ideal.corr()

wnom_model.ideal_points(torch.arange(0, 25))
wnom_model.w
pd.Series(y_pred.detach().numpy()).hist()
