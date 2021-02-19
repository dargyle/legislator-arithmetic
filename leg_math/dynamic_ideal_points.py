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


def process_data(vote_df, congress_cutoff=0, k_dim=1, k_time=0,
                 return_vote_df=False, validation_split=0.2, covariates_list=[],
                 unanimity_check=True):
    '''Process a dataframe of votes into a dictionary expected by the model

    # Arguments:
        vote_df (DataFrame): A DataFrame of votes to process
        congress_cutoff (int): drop votes from congresses prior to this number
        k_dim (int): the number of dimensions in the model
        k_time (int), EXPERIMENTAL: the number of time dimensions in the model
        return_vote_df (bool): return a data frame of the votes, in addition to
            the model data
        validation_split (float): percentage of the data to keep in the validation set
        covariates_list (list), EXPERIMENTAL: a list of covariate names to
            initialize addition of covariates to the model
    # Returns:
        vote_data (dict): a dictionary containing all the data necessary to fit
            the model
    '''
    logger.info("Limit the sample")
    if congress_cutoff:
        vote_df = vote_df[vote_df["congress"] >= congress_cutoff].copy()

    if k_time > 0:
        first_event = vote_df.groupby("leg_id")[["time_vote"]].agg(["min", "max"])
        first_event.columns = ["first_event", "last_event"]
        first_event["time_present"] = first_event["last_event"] - first_event["first_event"]
        # first_event["first_event"].value_counts()
        vote_df = pd.merge(vote_df, first_event, left_on="leg_id", right_index=True)
        vote_df["time_passed"] = vote_df["time_vote"] - vote_df["first_event"]
    else:
        first_event = pd.DataFrame(columns=["time_present"], index=vote_df["leg_id"].unique())

    # Shuffle the order of the vote data
    # THIS IS IMPORTANT, otherwise will_select just most recent bills
    vote_df = vote_df.sample(frac=1, replace=False, random_state=42)

    if "vote_weight" not in vote_df.columns:
        vote_df["vote_weight"] = 1.0

    N = len(vote_df)
    key_index = round(validation_split * N)
    logger.debug(f"key_index: {key_index}")

    # Keep only votes that are valid in the dataset
    train_data = vote_df.iloc[:(N - key_index), :].copy()
    if unanimity_check:
        train_data = drop_unanimous(train_data, min_vote_count=10, unanimity_percentage=0.001)
    # Ensure test data only contains valid entries
    test_data = vote_df.iloc[(N - key_index):, :]
    test_data = test_data[test_data["leg_id"].isin(train_data["leg_id"])]
    test_data = test_data[test_data["vote_id"].isin(train_data["vote_id"])]

    if k_time > 0:
        time_passed_train = [(train_data["time_passed"] ** i).values for i in range(0, k_time + 1)]
        time_passed_test = [(test_data["time_passed"] ** i).values for i in range(0, k_time + 1)]
    else:
        time_passed_train = []
        time_passed_test = []

    leg_ids = train_data["leg_id"].unique()
    vote_ids = train_data["vote_id"].unique()

    leg_crosswalk = pd.Series(leg_ids).to_dict()
    leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
    vote_crosswalk = pd.Series(vote_ids).to_dict()
    vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

    train_data["leg_id_num"] = train_data["leg_id"].map(leg_crosswalk_rev)
    train_data["vote_id_num"] = train_data["vote_id"].map(vote_crosswalk_rev)
    test_data["leg_id_num"] = test_data["leg_id"].map(leg_crosswalk_rev)
    test_data["vote_id_num"] = test_data["vote_id"].map(vote_crosswalk_rev)

    init_embedding = train_data[["leg_id_num", "init_value"]].drop_duplicates("leg_id_num").set_index("leg_id_num").sort_index()

    assert not vote_df.isnull().any().any(), "Missing value in data"

    vote_data = {'J': len(leg_ids),
                 'M': len(vote_ids),
                 'N': N,
                 'j_train': train_data[["leg_id_num"]].values,
                 'j_test': test_data[["leg_id_num"]].values,
                 'm_train': train_data[["vote_id_num"]].values,
                 'm_test': test_data[["vote_id_num"]].values,
                 'y_train': train_data[["vote"]].astype(int).values,
                 'y_test': test_data[["vote"]].astype(int).values,
                 'time_passed_train': time_passed_train,
                 'time_passed_test': time_passed_test,
                 'time_vote_train': train_data["time_vote"].values,
                 'time_vote_test': test_data["time_vote"].values,
                 'init_embedding': init_embedding,
                 'vote_crosswalk': vote_crosswalk,
                 'leg_crosswalk': leg_crosswalk,
                 'covariates_train': train_data[covariates_list].values,
                 'covariates_test': test_data[covariates_list].values,
                 'vote_weight_train': train_data["vote_weight"].values,
                 'vote_weight_test': test_data["vote_weight"].values,
                 'time_present': first_event.loc[leg_crosswalk_rev.keys(), ["first_event", "last_event", "time_present"]].values,
                 }

    # Export a pscl rollcall type object of the training data
    # if data_type == 'test':
    #     export_vote_df = vote_df.iloc[:(N - key_index), :]
    #     export_vote_df = export_vote_df[["leg_id", "vote_id", "vote"]]
    #     export_vote_df["leg_id"] = export_vote_df["leg_id"].map(leg_crosswalk)
    #     export_vote_df["vote_id"] = export_vote_df["vote_id"].map(vote_crosswalk)
    #     roll_call = export_vote_df.set_index(["leg_id", "vote_id"])["vote"].map({1: 1, 0: 6}).unstack()
    #     roll_call.fillna(9).astype(int).to_csv(DATA_PATH + "/test_votes.csv")

    init_leg_embedding_final = pd.DataFrame(np.random.uniform(-1.0, 1.0, size=(vote_data["J"], k_dim)))
    init_leg_embedding_final.iloc[:, 0] = init_leg_embedding_final.iloc[:, 0].abs() * vote_data["init_embedding"]["init_value"]
    max_norm = np.sqrt((init_leg_embedding_final ** 2).sum(axis=1)).max()
    init_leg_embedding_final = init_leg_embedding_final / (max_norm + 1e-7)

    vote_data['init_embedding'] = init_leg_embedding_final

    return vote_data

vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df["time_vote"] = vote_df["congress"]

k_dim = 1
k_time = 1
covariates_list = []
data_params = dict(
               congress_cutoff=110,
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
    covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.long, device=device)
if k_time > 0:
    time_tensor = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.long, device=device)
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

vote_data["time_present"]
first_session = torch.tensor(vote_data["time_present"][:, 0], dtype=torch.long, device=device)
last_session = torch.tensor(vote_data["time_present"][:, 1], dtype=torch.long, device=device)
time_vote = torch.tensor(vote_data["time_vote_train"], dtype=torch.long, device=device)
time_vote_test = torch.tensor(vote_data["time_vote_test"], dtype=torch.long, device=device)
sessions_served = last_session - first_session
ind_time_tensor = time_tensor[:, 1]
ind_time_tensor_test = time_tensor_test[:, 1]


# def random_walk(i, T):
#     x_0 = pyro.sample(f'x_0_{i}', dist.Normal(0, 1)).unsqueeze(-1)
#     sigma = pyro.sample(f'sigma_{i}', dist.LogNormal(0, 1)).unsqueeze(-1)
#     v = pyro.sample(f'v_{i}', dist.Normal(0, 1).expand([T]).to_event(1))
#     x = pyro.deterministic(f"x_{i}", x_0 + sigma * v.cumsum(dim=-1))
#     return x


def bayes_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, y=None, k_dim=1, device=None):
    """Define a core ideal point model

    Args:
        legs: a tensor of legislator ids
        votes: a tensor of vote ids
        y: a tensor of vote choices
        k_dim: desired dimensions of the models
    """
    # Set up parameter plates for all of the parameters
    # with pyro.plate('thetas', n_legs, dim=-2, device=device):
    #     ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))
    # ideal_point_dict = {i.item(): random_walk(i.item(), sessions_served[i] + 1) for i in legs.unique()}
    # nn = 0
    # n = len(legs)
    # ideal_point = torch.empty(n, 1)
    # for nn in range(n):
    #     temp_ideal = ideal_point_dict[legs[nn].item()]
    #     ideal_point[nn] = temp_ideal[ind_time_tensor[nn].item()]

    max_time = ind_time_tensor.max()

    with pyro.plate('thetas', n_legs, dim=-2, device=device):
        init_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('sigmas', n_legs, dim=-2, device=device):
        atten = pyro.sample('sigma', dist.LogNormal(torch.zeros(k_dim, device=device), 0.01 * torch.ones(k_dim, device=device)))

    with pyro.plate('vs', n_legs, dim=-2, device=device):
        disturb = pyro.sample('v', dist.Normal(torch.zeros(k_dim, max_time, device=device), torch.ones(k_dim, max_time, device=device)))

    with pyro.plate('betas', n_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', n_votes, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    later_ideal_point = init_ideal_point + (atten * disturb).squeeze().cumsum(dim=-1)
    ideal_point = torch.hstack([init_ideal_point, later_ideal_point])

    batch_ideal = ideal_point[legs, ind_time_tensor]

    # Combine parameters
    logit = torch.sum(batch_ideal.unsqueeze(-1) * polarity[votes], dim=-1) + popularity[votes]

    if y is not None:
        # If training outcomes are provided, run the sampling statement for the given data
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        # If no training outcomes are provided, return the samples from the predicted distributions
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


logger.info("Test a model with covariates")
logger.info("Setup the Bayesian Model")
# Choose the optimizer used by the variational algorithm
optim = Adam({'lr': 0.1})

# Define the guide, intialize to the values returned from process data
guide = autoguides.AutoNormal(bayes_irt_dynamic)
# guide = ideal_point_guide(legs, votes, responses, i)

# Setup the variational inference
svi = SVI(bayes_irt_dynamic, guide, optim, loss=Trace_ELBO())
# svi.step(legs, votes, responses, covariates, k_dim=k_dim)

logger.info("Run the variational inference")
pyro.clear_param_store()
for j in tqdm(range(5000)):
    loss = svi.step(legs, n_legs, votes, n_votes, ind_time_tensor, y=responses, k_dim=1, device=None)
    if j % 100 == 0:
        logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
init_ideal_point = pyro.param("AutoNormal.locs.theta")
atten = pyro.param("AutoNormal.locs.sigma")
atten.shape
disturb = pyro.param("AutoNormal.locs.v")
disturb.shape

init_ideal_point.shape
later_ideal_point = init_ideal_point + (atten * disturb).squeeze().cumsum(dim=-1)
later_ideal_point.shape
ideal_point = torch.hstack([init_ideal_point, later_ideal_point])
ideal_point.shape

indices = (torch.arange(start=0, end=max_time + 1).repeat(len(ideal_point), 1))
mask = indices.le(sessions_served.unsqueeze(-1))

ideal_point * mask
