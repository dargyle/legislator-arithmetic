import pyro
import pyro.distributions as dist
import torch

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides
from torch.distributions import constraints

from pyro.infer.autoguide.initialization import init_to_value

import pandas as pd
import numpy as np

# from data_generation.data_processing import process_data
from data_generation.data_processing import drop_unanimous, format_model_data

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

from leg_math.pytorch_bayes import bayes_irt_basic, bayes_irt_full, normalize_ideal_points

from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

pyro.enable_validation(True)

# Set up environment
gpu = False

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["congress"] == 115]
vote_df = vote_df[vote_df["chamber"] == "Senate"]

vote_df["vote_new"] = vote_df["vote"].replace({0: -1})
vote_mean = vote_df.groupby("vote_id")["vote_new"].mean()
vote_mean.name = "vote_mean"
leg_mean = vote_df.groupby("leg_id")["vote_new"].mean()
leg_mean.name = "leg_mean"
overall_mean = vote_df["vote_new"].mean()

vote_df = pd.merge(vote_df, vote_mean, left_on="vote_id", right_index=True)
vote_df = pd.merge(vote_df, leg_mean, left_on="leg_id", right_index=True)
vote_df["init_value"] = vote_df["vote_new"] - vote_df["vote_mean"] - vote_df["leg_mean"] + overall_mean

asdf = vote_df[["leg_id", "vote_id", "vote_new"]].set_index(["leg_id", "vote_id"])
roll_call = asdf.unstack()["vote_new"].fillna(0)
row_means = roll_call.mean(axis=1)
col_means = roll_call.mean(axis=0)
all_mean = roll_call.mean()
roll_call = roll_call.sub(row_means, axis=0)
roll_call = roll_call.sub(col_means, axis=1)
roll_call = roll_call + overall_mean

from scipy.linalg import svd

u, s, v = svd(roll_call.values, full_matrices=False)
u.shape
s.shape
v.shape
(u @ np.diag(s) @ v)
k_dim = 3

for k_dim in range(107):
    roll_call_approx = (u[:, :k_dim]  @ np.diag(s[:k_dim]) @ v[:k_dim, :])
    print(np.mean((roll_call.values.flatten() - roll_call_approx.flatten()) ** 2))

leg_ids = vote_df["leg_id"].unique()
vote_ids = vote_df["vote_id"].unique()

leg_crosswalk = pd.Series(leg_ids).to_dict()
leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
vote_crosswalk = pd.Series(vote_ids).to_dict()
vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

vote_df["leg_id_num"] = vote_df["leg_id"].map(leg_crosswalk_rev)
vote_df["vote_id_num"] = vote_df["vote_id"].map(vote_crosswalk_rev)

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
coo_matrix(vote_df["init_value"].values, (vote_df["leg_id_num"].values, vote_df["vote_id_num"].values))


init_series = vote_df[["leg_id", "vote_id", "init_value"]].set_index(["leg_id", "vote_id"])["init_value"]
init_series_sparse = init_series.astype("Sparse")
A, rows, columns = init_series_sparse.sparse.to_coo(row_levels=["leg_id"], column_levels=["vote_id"])

A.todense()

u, s, v = svds(A, k=2)
u @ (s * np.eye(len(s))) @ v

df = vote_df.set_index(["leg_id", "vote_id"])["vote"]
roll_call = df.unstack("vote_id")
roll_call = roll_call.replace({0: -1})
# roll_call = roll_call.fillna(0)

vote_mean = roll_call.mean(axis=0)
leg_mean = roll_call.mean(axis=1)

roll_call = roll_call.sub(vote_mean, axis=1)
roll_call = roll_call.sub(leg_mean, axis=0)

roll_call
roll_call




vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")

k_dim = 2
k_time = 0
covariates_list = []

from data_generation.data_processing import process_data, format_model_data

data_params = dict(
               congress_cutoff=0,
               k_dim=k_dim,
               k_time=k_time,
               covariates_list=covariates_list,
               unanimity_check=True,
               )
vote_data = process_data(vote_df=vote_df, **data_params)
custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

# Convert training and test data to tensors
legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
if covariates_list:
    covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
if k_time > 0:
    time_tensor = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)
    time_present = torch.tensor(vote_data["time_present"], device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
if covariates_list:
    covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
if k_time > 0:
    time_tensor_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# Set some constants
n_legs = vote_data["J"]
n_votes = vote_data["M"]
if covariates_list:
    n_covar = covariates.shape[1]



from surprise import Reader, Dataset
vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
# vote_df = vote_df[vote_df["congress"] >= 110]
reader = Reader(rating_scale=(0.0, 1.0))
data = Dataset.load_from_df(vote_df[['leg_id', 'vote_id', 'vote']], reader)
trainset = data.build_full_trainset()

# Split data into 5 folds

# data.split(n_folds=5)

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# svd
algo = SVD()
algo.fit(trainset)
predict_list = algo.test(trainset.build_testset(), verbose=False)
predict_list[0]

algo.__dict__
algo.pu.shape
algo.qi.shape

vote_df
len(vote_df["leg_id"].unique())

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=True, n_jobs=1)


# nmf
algo = NMF()
evaluate(algo, data, measures=['RMSE'])


import torch
import torch.nn as nn

class cf_baseline(nn.Module):

    def __init__(self, n_legs, n_votes, k_dim=1):
        super().__init__()
        # self.theta = nn.Embedding(n_legs, k_dim)
        self.theta = nn.Parameter(torch.randn(n_legs, k_dim))
        # self.beta = nn.Embedding(n_votes, k_dim)
        self.beta = nn.Parameter(torch.randn(n_votes, k_dim))
        # self.theta.weight.data.uniform_(0, 0.05)
        # self.beta.weight.data.uniform_(0, 0.05)

        self.theta_mean = nn.Parameter(torch.randn(n_legs))
        self.beta_mean = nn.Parameter(torch.randn(n_votes))
        self.overall_mean = nn.Parameter(torch.zeros(1))

    def forward(self, legs, votes):
        # temp_sum = torch.sum(self.theta(legs) * self.beta(votes), axis=1)
        temp_sum = torch.sum(self.theta[legs] * self.beta[votes], axis=1)
        return temp_sum + self.theta_mean[legs] + self.beta_mean[votes] + self.overall_mean


cf_model = cf_baseline(n_legs, n_votes, k_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(cf_model.parameters(), amsgrad=True)

logger.info("Fit the pytorch model")
losses = []
accuracies = []
test_losses = []
test_accuracies = []
for t in tqdm(range(5000)):
    y_pred = cf_model(legs, votes)
    loss = criterion(y_pred, responses)

    with torch.no_grad():
        accuracy = ((y_pred > 0) == responses).sum().item() / len(responses)

        y_pred_test = cf_model(legs_test, votes_test)
        loss_test = criterion(y_pred_test, responses_test)
        accuracy_test = ((y_pred_test > 0) == responses_test).sum().item() / len(responses_test)

    if t % 100 == 0:
        logger.info(f'epoch {t}, loss: {loss.item()}')

    losses.append(loss.item())
    accuracies.append(accuracy)
    test_losses.append(loss_test.item())
    test_accuracies.append(accuracy_test)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

n_votes
