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

from tqdm import tqdm

pyro.enable_validation(True)


# Set up environment
# Untested with GPU, that's part of what this is about
gpu = False

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Add a covariate
random_votes = generate_nominate_votes(n_leg=100, n_votes=1000, beta=15.0, beta_covar=0.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
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
    def __init__(self, num_legs, num_votes, k_dim, pretrained=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom, self).__init__()
        # self.ideal_points = nn.Embedding.from_pretrained(pretrained, max_norm=1.0)
        # self.yes_points = nn.Embedding(num_embeddings=num_votes, embedding_dim=k_dim)
        # nn.init.uniform_(self.yes_points.weight)
        # self.no_points = nn.Embedding(num_embeddings=num_votes, embedding_dim=k_dim)
        # nn.init.uniform_(self.no_points.weight)

        if pretrained is not None:
            self.ideal_points = nn.Parameter(pretrained)
        else:
            self.ideal_points = nn.Parameter(torch.rand(num_legs, k_dim))
        self.yes_points = nn.Parameter(torch.rand(num_votes, k_dim))
        self.no_points = nn.Parameter(torch.rand(num_votes, k_dim))

        self.w = nn.Parameter(0.5 * torch.ones((k_dim)))
        self.beta = nn.Parameter(torch.tensor(5.0))
        # self.sig = nn.Sigmoid()

    def max_norm_(self, w):
        with torch.no_grad():
            norm = w.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1)
            w *= desired / norm

    def forward(self, legs, votes):
        """
        Take in the legislator and vote ids and generate a prediction
        """
        # distances1 = torch.sum(torch.square(self.ideal_points(legs) - self.yes_points(votes)) * torch.square(self.w), axis=1)
        # distances2 = torch.sum(torch.square(self.ideal_points(legs) - self.no_points(votes)) * torch.square(self.w), axis=1)

        # Constrain all these things to the a unit hypersphere (ala original NOMINATE)
        self.max_norm_(self.ideal_points)
        self.max_norm_(self.yes_points)
        self.max_norm_(self.no_points)

        distances1 = torch.sum(torch.square(self.ideal_points[legs] - self.yes_points[votes]) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(self.ideal_points[legs] - self.no_points[votes]) * torch.square(self.w), axis=1)

        result = self.beta * (torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))
        # result = self.sig(torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


wnom_model = wnom(num_legs, num_votes, k_dim, custom_init_values)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)

losses = []
accuracies = []
test_losses = []
test_accuracies = []
for t in tqdm(range(5000)):
    y_pred = wnom_model(legs, votes)
    loss = criterion(y_pred, responses)

    with torch.no_grad():
        # accuracy = accuracy_score(responses, y_pred.numpy() >= 0.5)
        accuracy = ((y_pred > 0) == responses).sum().item() / len(responses)

        y_pred_test = wnom_model(legs_test, votes_test)
        loss_test = criterion(y_pred_test, responses_test)
        # accuracy_test = accuracy_score(responses_test, y_pred_test.numpy() >= 0.5)
        accuracy_test = ((y_pred_test > 0) == responses_test).sum().item() / len(responses_test)

    # if t % 100 == 99:
    #     print(t, loss.item())

    losses.append(loss.item())
    accuracies.append(accuracy)
    test_losses.append(loss_test.item())
    test_accuracies.append(accuracy_test)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)

(wnom_model.ideal_points[torch.arange(0, 100)] ** 2).sum(dim=1)
wnom_model.ideal_points[torch.arange(0, 100)].norm(dim=1)

accuracy_score(responses, 1 * wnom_model(legs, votes).detach().numpy() >= 0.5)

pd.Series(losses).plot()
pd.Series(test_losses).plot()

pd.Series(accuracies).plot()
pd.Series(test_accuracies).plot()


true_ideal = random_votes[["coord1D", "coord2D"]]
true_ideal.index = true_ideal.index.droplevel("vote_id")
true_ideal = true_ideal.drop_duplicates()
vote_data.keys()
leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}
true_ideal[["wnom1", "wnom2"]] = wnom_model.ideal_points[torch.tensor(true_ideal.index.map(leg_crosswalk_rev).values)].detach().numpy()

true_ideal.corr()
true_ideal.plot(kind='scatter', x="coord1D", y="wnom1")
true_ideal.plot(kind='scatter', x="coord2D", y="wnom2")

X = wnom_model.ideal_points[torch.arange(0, 100)].detach()
Y = torch.tensor(true_ideal[["coord1D", "coord2D"]].values, dtype=torch.float)

ab = torch.inverse(X.transpose(0, 1).mm(X))
cd = X.transpose(0, 1).mm(Y)
rot = ab.mm(cd)

from scipy.linalg import orthogonal_procrustes
rot, _ = orthogonal_procrustes(X, Y)
temp_X = X.mm(torch.tensor(rot))

true_ideal[["wnom1", "wnom2"]] = temp_X.numpy()
true_ideal.corr()

pd.DataFrame(temp_X.numpy()).plot(kind='scatter', x=0, y=1)

(wnom_model.yes_points[torch.arange(0, 100)] ** 2).sum(dim=1).max()
pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].detach().numpy()).plot(kind='scatter', x=0, y=1)
wnom_model.w
wnom_model.beta
pd.Series(y_pred.detach().numpy()).hist()


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

preds_list = []
for _ in range(1000):
    guide_trace = pyro.poutine.trace(guide).get_trace(legs, votes)
    preds_list.append(pyro.poutine.replay(ideal_point_model, guide_trace)(legs, votes))
preds = torch.stack(preds_list)

pd.concat([pd.Series(preds.mean(axis=0).numpy()),
           pd.Series(nn.Sigmoid()(wnom_model(legs, votes)).detach().numpy())],
          axis=1).plot(kind="scatter", x=0, y=1)
temp_loss = torch.nn.BCELoss()
temp_loss(preds.mean(axis=0), responses)
accuracy_score(responses, 1 * (preds.mean(axis=0) >= 0.5))







# Add a covariate
random_votes = generate_nominate_votes(n_leg=100, n_votes=1000, beta=15.0, beta_covar=0.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
vote_df = random_votes.reset_index()

k_dim = 2
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
               k_dim=k_dim,
               k_time=1,
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

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
# covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# Set some constants
num_legs = len(set(legs.numpy()))
num_votes = len(set(votes.numpy()))
# num_covar = covariates.shape[1]


class wnom_time(nn.Module):
    def __init__(self, num_legs, num_votes, k_dim, pretrained=None, k_time=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom_time, self).__init__()
        if k_time is not None:
            self.ideal_points = nn.Parameter((torch.rand(num_legs, k_dim, k_time + 1) - 0.5) / 5.0)
        else:
            if pretrained is not None:
                self.ideal_points = nn.Parameter(pretrained)
            else:
                self.ideal_points = nn.Parameter(torch.rand(num_legs, k_dim))
        self.yes_points = nn.Parameter(torch.rand(num_votes, k_dim))
        self.no_points = nn.Parameter(torch.rand(num_votes, k_dim))

        self.w = nn.Parameter(0.5 * torch.ones((k_dim)))
        self.beta = nn.Parameter(torch.tensor(5.0))
        # self.sig = nn.Sigmoid()

    def max_norm_(self, w):
        with torch.no_grad():
            norm = w.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1)
            w *= desired / norm

    def reduced_max_norm_(self, w):
        with torch.no_grad():
            reduced_w = w.sum(dim=2)
            norm = reduced_w.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1)
            w *= (desired / norm).unsqueeze(dim=-1)

    def forward(self, legs, votes, time_tensor=None):
        """
        Take in the legislator and vote ids and generate a prediction
        """
        # distances1 = torch.sum(torch.square(self.ideal_points(legs) - self.yes_points(votes)) * torch.square(self.w), axis=1)
        # distances2 = torch.sum(torch.square(self.ideal_points(legs) - self.no_points(votes)) * torch.square(self.w), axis=1)

        # Constrain all these things to the a unit hypersphere (ala original NOMINATE)
        self.reduced_max_norm_(self.ideal_points)
        self.max_norm_(self.yes_points)
        self.max_norm_(self.no_points)

        if time_tensor is not None:
            ideal_points_use = torch.sum(self.ideal_points[legs] * time_tensor.unsqueeze(-1), axis=1)

            # with torch.no_grad():
            #     norm = ideal_points_use.norm(2, dim=1, keepdim=True)
            #     desired = torch.clamp(norm, max=1)
            #     ideal_points_use *= desired / norm
            # self.max_norm_(ideal_points_use)
        else:
            self.max_norm_(self.ideal_points)
            ideal_points_use = self.ideal_points

        distances1 = torch.sum(torch.square(ideal_points_use[legs] - self.yes_points[votes]) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(ideal_points_use[legs] - self.no_points[votes]) * torch.square(self.w), axis=1)

        result = self.beta * (torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))
        # result = self.sig(torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


wnom_model = wnom_time(num_legs, num_votes, k_dim, pretrained=custom_init_values, k_time=1)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)

time_tensor = torch.cat([torch.ones(time_passed.shape[0], device=device).unsqueeze(-1), time_passed], axis=1)
time_tensor_test = torch.cat([torch.ones(time_passed_test.shape[0], device=device).unsqueeze(-1), time_passed_test], axis=1)

losses = []
accuracies = []
test_losses = []
test_accuracies = []
for t in tqdm(range(5000)):
    y_pred = wnom_model(legs, votes, time_tensor)
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
    loss.backward(retain_graph=True)
    optimizer.step()
    # print(loss)

wnom_model.ideal_points[torch.arange(0, 100)]
time_tensor.unsqueeze(-1).shape
wnom_model.ideal_points[legs] * time_tensor.unsqueeze(-1)
torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(-1), axis=1)
torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(-1), axis=1).norm(dim=1)

wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)

pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
