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

k_dim = 2
k_time = 1
covariates_list = []
data_params = dict(
               congress_cutoff=93,
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
    time_tensor_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.long, device=device)

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

max_time = ind_time_tensor.max()


import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item


class custom_SVI(object):
    """
    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.Trace_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.Trace_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :type loss: pyro.infer.elbo.ELBO

    A unified interface for stochastic variational inference in Pyro. The most
    commonly used loss is ``loss=Trace_ELBO()``. See the tutorial
    `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 l2_penalty=1.0,
                 **kwargs):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.l2_penalty = l2_penalty

        if isinstance(loss, ELBO):
            self.loss = loss.loss
            self.loss_and_grads = loss.loss_and_grads
        else:
            if loss_and_grads is None:
                def _loss_and_grads(*args, **kwargs):
                    loss_val = loss(*args, **kwargs)
                    # loss_val.backward()
                    return loss_val
                loss_and_grads = _loss_and_grads
            self.loss = loss
            self.loss_and_grads = loss_and_grads

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            return torch_item(self.loss(self.model, self.guide, *args, **kwargs))

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        return torch_item(loss)

    def my_custom_L2_regularizer(self, my_parameters):
        reg_loss = 0.0
        for param in my_parameters:
            reg_loss = reg_loss + self.l2_penalty * param.pow(2.0).sum()
        return reg_loss

    def penalized_step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())

        # define optimizer and loss function
        # optimizer = torch.optim.Adam(params, {"lr": 0.1})
        # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        # compute loss
        # loss = loss_fn(self.model, self.guide, *args, **kwargs) + self.my_custom_L2_regularizer(params)
        l2_loss = self.my_custom_L2_regularizer(params)
        # print(loss)
        # print(l2_loss)
        final_loss = loss + l2_loss
        # print(final_loss)
        final_loss.backward()
        # take a step and zero the parameter gradients
        # self.optim.step()
        # self.optim.zero_grad()
        self.optim(params)
        # zero gradients
        pyro.infer.util.zero_grads(params)

        return torch_item(final_loss)


class bayes_dynamic:
    def __init__(self, n_legs, n_votes, max_time, k_dim=1, guide_type="auto", custom_init_values=None, device=None):
        super().__init__()
        self.n_legs = n_legs
        self.n_votes = n_votes
        self.max_time = max_time
        self.k_dim = k_dim
        self.guide_type = guide_type
        self.custom_init_values = custom_init_values
        self.device = device

    def model(self, legs, votes, ind_time_tensor, obs=None):
        with pyro.plate('thetas', n_legs, dim=-2, device=device):
            init_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(self.k_dim, device=self.device), torch.ones(self.k_dim, device=self.device)))

        with pyro.plate('sigmas', self.n_legs, dim=-2, device=self.device):
            # atten = pyro.sample('sigma', dist.Exponential(1.0 * torch.ones(self.k_dim, device=self.device)))
            # atten = pyro.sample('sigma', dist.Normal(torch.zeros(self.k_dim, device=self.device), torch.ones(self.k_dim, device=self.device)))
            atten = pyro.sample('sigma', dist.Gamma(torch.ones(self.k_dim, device=self.device), torch.ones(self.k_dim, device=self.device)))

        with pyro.plate('vs', self.n_legs, dim=-3, device=self.device):
            disturb = pyro.sample('v', dist.Normal(torch.zeros(self.k_dim, max_time, device=self.device), torch.ones(self.k_dim, max_time, device=self.device)))

        with pyro.plate('betas', self.n_votes, dim=-2,  device=self.device):
            polarity = pyro.sample('beta', dist.Normal(torch.zeros(self.k_dim, device=self.device), 5.0 * torch.ones(self.k_dim, device=self.device)))

        with pyro.plate('alphas', self.n_votes, dim=-2, device=self.device):
            popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=self.device), 5.0 * torch.ones(1, device=self.device)))

        temp_ideal = init_ideal_point.unsqueeze(-1)
        # print(temp_ideal.shape)
        # print(atten.shape)
        # print(disturb.shape)
        all_temp_ideal = temp_ideal + (atten.unsqueeze(-1) * disturb).cumsum(dim=-1)
        # print(all_temp_ideal.shape)

        ideal_point = pyro.deterministic("ideal_point", torch.cat([temp_ideal, all_temp_ideal], dim=-1))
        # ideal_point.shape

        batch_ideal = ideal_point[legs, :, ind_time_tensor]
        polarity_batch = polarity[votes]
        popularity_batch = popularity[votes]

        # batch_ideal.shape
        # polarity_batch.shape
        # popularity_batch.shape

        # Combine parameters
        logit = torch.sum(batch_ideal * polarity_batch, dim=-1) + popularity_batch.squeeze()

        if obs is not None:
            # If training outcomes are provided, run the sampling statement for the given data
            with pyro.plate('observe_data', obs.size(0), device=self.device):
                pyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)
        else:
            # If no training outcomes are provided, return the samples from the predicted distributions
            with pyro.plate('observe_data', legs.size(0), device=self.device):
                obs = pyro.sample("obs", dist.Bernoulli(logits=logit))
            return obs

    def guide_basic(self, legs, votes, ind_time_tensor, obs=None):
        # register parameters
        loc_theta = pyro.param("loc_init_ideal", torch.zeros(self.n_legs, self.k_dim, device=self.device))
        scale_theta = pyro.param("scale_init_ideal", torch.ones(self.n_legs, self.k_dim, device=self.device),
                                 constraint=constraints.positive)
        loc_sigma = pyro.param("loc_atten", torch.ones(self.n_legs, self.k_dim, device=device),
                               constraint=constraints.positive)
        scale_sigma = pyro.param("scale_atten", torch.ones(self.n_legs, self.k_dim, device=self.device),
                                 constraint=constraints.positive)
        loc_v = pyro.param("loc_disturb", torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
        scale_v = pyro.param("scale_disturb", torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device),
                             constraint=constraints.positive)
        loc_beta = pyro.param("loc_polarity", torch.zeros(self.n_votes, self.k_dim, device=self.device))
        scale_beta = pyro.param("scale_polarity", 5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device),
                                constraint=constraints.positive)
        loc_alpha = pyro.param("loc_popularity", torch.zeros(self.n_votes, 1, device=self.device))
        scale_alpha = pyro.param("scale_popularity", 5.0 * torch.ones(self.n_votes, 1, device=self.device),
                                 constraint=constraints.positive)

        with pyro.plate('thetas', n_legs, dim=-2, device=self.device):
            pyro.sample('theta', dist.Normal(loc_theta, scale_theta))
        with pyro.plate('sigmas', n_legs, dim=-2, device=self.device):
            pyro.sample('sigma', dist.Gamma(loc_sigma, scale_sigma))
        with pyro.plate('vs', n_legs, dim=-3, device=self.device):
            pyro.sample('v', dist.Normal(loc_v, scale_v))
        with pyro.plate('betas', n_votes, dim=-2, device=self.device):
            pyro.sample('beta', dist.Normal(loc_beta, scale_beta))
        with pyro.plate('alphas', n_votes, dim=-2, device=self.device):
            pyro.sample('alpha', dist.Normal(loc_alpha, scale_alpha))

    def fit(self, legs, votes, ind_time_tensor, responses):
        if self.guide_type == "auto":
            guide = autoguides.AutoNormal(self.model)
        elif self.guide_type == "basic":
            guide = self.guide_basic
            # Define the guide, intialize to the values returned from process data
            # guide = autoguides.AutoNormal(bayes_irt_dynamic)
            # guide = guide_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
            # guide = bayes_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, k_dim=1, device=None)
        pyro.clear_param_store()
        if self.guide_type == "auto":
            pyro.get_param_store().setdefault("AutoNormal.locs.sigma", init_constrained_value=1.0 * torch.ones(self.n_legs, self.k_dim), constraint=constraints.positive)
            pyro.get_param_store().setdefault("AutoNormal.locs.theta", init_constrained_value=self.custom_init_values)
            pyro.get_param_store().setdefault("AutoNormal.locs.v", init_constrained_value=torch.zeros(self.n_legs, self.k_dim, self.max_time))
        elif self.guide_type == "basic":
            pyro.get_param_store().setdefault("loc_init_ideal", init_constrained_value=self.custom_init_values)
            pyro.get_param_store().setdefault("scale_init_ideal", init_constrained_value=torch.ones(self.n_legs, self.k_dim), constraint=constraints.positive)
            pyro.get_param_store().setdefault("loc_atten", init_constrained_value=0.01 * torch.ones(self.n_legs, self.k_dim), constraint=constraints.positive)
            pyro.get_param_store().setdefault("scale_atten", init_constrained_value=0.01 * torch.ones(self.n_legs, self.k_dim), constraint=constraints.positive)
            pyro.get_param_store().setdefault("loc_disturb", init_constrained_value=torch.zeros(self.n_legs, self.k_dim, self.max_time))
            pyro.get_param_store().setdefault("scale_disturb", init_constrained_value=torch.ones(self.n_legs, self.k_dim, self.max_time), constraint=constraints.positive)
            pyro.get_param_store().setdefault("loc_beta", init_constrained_value=torch.zeros(self.n_votes, self.k_dim, device=self.device))
            pyro.get_param_store().setdefault("scale_beta", init_constrained_value=5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device), constraint=constraints.positive)
            pyro.get_param_store().setdefault("loc_beta", init_constrained_value=torch.zeros(self.n_votes, 1, device=self.device))
            pyro.get_param_store().setdefault("scale_beta", init_constrained_value=5.0 * torch.ones(self.n_votes, 1, device=self.device), constraint=constraints.positive)

        logger.info("Run the variational inference")
        # Setup the variational inference
        optim = Adam({'lr': 0.1})
        svi = custom_SVI(self.model, guide, optim, loss=Trace_ELBO(), l2_penalty=0.0)

        min_loss = float('inf')  # initialize to infinity
        patience = 0
        for j in tqdm(range(5000)):
            # loss = svi.step(legs, votes, ind_time_tensor, responses)
            loss = svi.penalized_step(legs, votes, ind_time_tensor, responses)

            if j % 100 == 0:
                logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
                min_loss = min(loss, min_loss)
                if (loss > min_loss):
                    if patience >= 5:
                        break
                    else:
                        patience += 1
                else:
                    patience = 0


pyro.clear_param_store()
test = bayes_dynamic(n_legs, n_votes, max_time, k_dim=k_dim, guide_type="basic", custom_init_values=custom_init_values, device=device)
test.fit(legs, votes, ind_time_tensor, responses)

posterior_predictive = pyro.infer.predictive.Predictive(model=test.model, guide=test.guide_basic, num_samples=1, return_sites=["obs", "ideal_point"])

preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
ideal_point_list = []
n_samples = 1000
for _ in tqdm(range(n_samples)):
    guide_trace = pyro.poutine.trace(test.guide_basic).get_trace(legs, votes, ind_time_tensor)
    posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(test.model, guide_trace)).get_trace(legs, votes, ind_time_tensor)
    preds += posterior_predictive.nodes['obs']['value'].squeeze()
    ideal_point_list += [posterior_predictive.nodes["ideal_point"]['value'].squeeze()]

    guide_trace = pyro.poutine.trace(test.guide_basic).get_trace(legs_test, votes_test, ind_time_tensor_test)
    posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(test.model, guide_trace)).get_trace(legs_test, votes_test, ind_time_tensor_test)
    preds_test += posterior_predictive.nodes['obs']['value'].squeeze()
preds = preds / n_samples
preds_test = preds_test / n_samples

# Define metrics
criterion = torch.nn.BCELoss(reduction="mean")
log_like = torch.nn.BCELoss(reduction="sum")
k = 0
for param_name, param_value in pyro.get_param_store().items():
    if "locs" in param_name:
        k += np.array(param_value.shape).prod()

# Calculate log loss and accuracy scores, in sample
train_metrics = {"bce": criterion(preds, responses).item(),
                 "log_like": log_like(preds, responses).item(),
                 "accuracy": (1.0 * (responses == preds.round())).mean().item(),
                }
logger.info(f'Train loss (VB): {train_metrics["bce"]}')
logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

# Calculate log loss and accuracy scores, out of sample
test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                "log_like": log_like(preds_test, responses_test).item(),
                "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(),
                }
logger.info(f'Test loss (VB): {test_metrics["bce"]}')
logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

metrics = {**train_metrics, **test_metrics}

metrics["train_n"] = responses.shape[0]
metrics["train_k"] = k
metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

metrics["test_n"] = responses_test.shape[0]
metrics["test_k"] = k
metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])

final_metrics = {**data_params, **metrics}
final_metrics
final_metrics



pyro.get_param_store().keys()
pyro.get_param_store()["loc_init_ideal"]
pyro.get_param_store()["loc_atten"]
pyro.get_param_store()["scale_atten"]
pyro.get_param_store()["loc_disturb"]
# pyro.get_param_store().named_parameters()

indices = (torch.arange(start=0, end=max_time + 1).repeat(n_legs, 1))
mask = indices.le(sessions_served.unsqueeze(-1))
mask * posterior_predictive.nodes['ideal_point']['value'].squeeze()

torch.stack(ideal_point_list, dim=-1).mean(axis=2)
torch.stack(ideal_point_list, dim=-1).mean(axis=2) * mask
torch.stack(ideal_point_list, dim=-1).std(axis=2) * mask
ideal_df = pd.DataFrame((torch.stack(ideal_point_list, dim=-1).mean(axis=2) * mask).detach().numpy())

leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

# leg_data[leg_data["leg_id"] == vote_data["leg_crosswalk"][1065]]
# ideal_df.loc[1065, :]

leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}

leg_data[leg_data["bioname"].str.contains("FLAKE")]
leg_data[leg_data["bioname"].str.contains("McCAIN")]
leg_data[leg_data["bioname"].str.contains("GILLIBRAND")]

ideal_df.loc[leg_crosswalk_rev[20100]]
ideal_df.loc[leg_crosswalk_rev[15039]]
ideal_df.loc[leg_crosswalk_rev[20735]]

# Define metrics
criterion = torch.nn.BCELoss(reduction="mean")
log_like = torch.nn.BCELoss(reduction="sum")
k = 0
for param_name, param_value in pyro.get_param_store().items():
    if "locs" in param_name:
        k += np.array(param_value.shape).prod()

# Calculate log loss and accuracy scores, in sample
train_metrics = {"bce": criterion(preds, responses).item(),
                 "log_like": log_like(preds, responses).item(),
                 "accuracy": (1.0 * (responses == preds.round())).mean().item(),
                }
logger.info(f'Train loss (VB): {train_metrics["bce"]}')
logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

# Calculate log loss and accuracy scores, out of sample
test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                "log_like": log_like(preds_test, responses_test).item(),
                "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(),
                }
logger.info(f'Test loss (VB): {test_metrics["bce"]}')
logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

metrics = {**train_metrics, **test_metrics}

metrics["train_n"] = responses.shape[0]
metrics["train_k"] = k
metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

metrics["test_n"] = responses_test.shape[0]
metrics["test_k"] = k
metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])

final_metrics = {**data_params, **metrics}
final_metrics


n_samples = 100
preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
ideal_point_list = []
for _ in tqdm(range(n_samples)):
    in_sample_predictive = posterior_predictive(legs, votes, ind_time_tensor)
    preds += in_sample_predictive["obs"].squeeze()
    ideal_point_list += [in_sample_predictive["ideal_point"].squeeze()]
    preds_test += posterior_predictive(legs_test, votes_test, ind_time_tensor_test)["obs"].squeeze()
preds = preds / n_samples
preds_test = preds_test / n_samples

ideal_point_list[10]
torch.stack(ideal_point_list, dim=-1).mean(axis=2) * mask
ideal_point * mask
torch.stack(ideal_point_list, dim=-1).std(axis=2) * masktest






def bayes_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=None, k_dim=1, device=None):
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

    with pyro.plate('thetas', n_legs, dim=-2, device=device):
        init_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('sigmas', n_legs, dim=-2, device=device):
        atten = pyro.sample('sigma', dist.Exponential(1.0 * torch.ones(k_dim, device=device)))
        # atten = pyro.sample('sigma', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

    with pyro.plate('vs', n_legs, dim=-3, device=device):
        disturb = pyro.sample('v', dist.Normal(torch.zeros(k_dim, max_time, device=device), torch.ones(k_dim, max_time, device=device)))

    with pyro.plate('betas', n_votes, dim=-2,  device=device):
        polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

    with pyro.plate('alphas', n_votes, dim=-2, device=device):
        popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))

    atten.shape
    disturb.shape
    (atten.unsqueeze(-1) * disturb).shape
    ideal_point = pyro.deterministic("ideal_point", torch.hstack([init_ideal_point, init_ideal_point + (atten.unsqueeze(-1) * disturb).squeeze().cumsum(dim=-1)]))
    # later_ideal_point = init_ideal_point + (atten * disturb).squeeze().cumsum(dim=-1)
    # ideal_point = torch.hstack([init_ideal_point, later_ideal_point])

    batch_ideal = ideal_point[legs, ind_time_tensor].unsqueeze(-1)
    polarity_batch = polarity[votes]
    popularity_batch = popularity[votes]

    # Combine parameters
    logit = torch.sum(batch_ideal * polarity_batch, dim=-1) + popularity_batch.squeeze()

    if y is not None:
        # If training outcomes are provided, run the sampling statement for the given data
        with pyro.plate('observe_data', y.size(0), device=device):
            pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
    else:
        # If no training outcomes are provided, return the samples from the predicted distributions
        with pyro.plate('observe_data', legs.size(0), device=device):
            y = pyro.sample("obs", dist.Bernoulli(logits=logit))
        return y


def guide_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=None, k_dim=1, device=None):
    # register parameters
    loc_theta = pyro.param("loc_init_ideal", torch.zeros(k_dim, device=device))
    scale_theta = pyro.param("scale_init_ideal", torch.ones(k_dim, device=device),
                             constraint=constraints.positive)
    # loc_sigma = pyro.param("loc_atten", torch.zeros(k_dim, device=device))
    scale_sigma = pyro.param("scale_atten", torch.ones(k_dim, device=device),
                             constraint=constraints.positive)
    loc_v = pyro.param("loc_disturb", torch.zeros(k_dim, max_time, device=device))
    scale_v = pyro.param("scale_disturb", torch.ones(k_dim, max_time, device=device),
                         constraint=constraints.positive)
    loc_beta = pyro.param("loc_polarity", torch.zeros(k_dim, device=device))
    scale_beta = pyro.param("scale_polarity", 5 * torch.ones(k_dim, device=device),
                            constraint=constraints.positive)
    loc_alpha = pyro.param("loc_popularity", torch.zeros(1, device=device))
    scale_alpha = pyro.param("scale_popularity", 5 * torch.ones(1, device=device),
                             constraint=constraints.positive)

    with pyro.plate('thetas', n_legs, dim=-2, device=device):
        pyro.sample('theta', dist.Normal(loc_theta, scale_theta))
    with pyro.plate('sigmas', n_legs, dim=-2, device=device):
        pyro.sample('sigma', dist.Exponential(scale_sigma))
    with pyro.plate('vs', n_legs, dim=-3, device=device):
        pyro.sample('v', dist.Normal(loc_v, scale_v))
    with pyro.plate('betas', n_votes, dim=-2, device=device):
        pyro.sample('beta', dist.Normal(loc_beta, scale_beta))
    with pyro.plate('alphas', n_votes, dim=-2, device=device):
        pyro.sample('alpha', dist.Normal(loc_alpha, scale_alpha))

pyro.clear_param_store()


logger.info("Test a model with covariates")
logger.info("Setup the Bayesian Model")
# Choose the optimizer used by the variational algorithm
optim = Adam({'lr': 0.1, 'weight_decay': 0.00001})

# Define the guide, intialize to the values returned from process data
guide = autoguides.AutoNormal(bayes_irt_dynamic)
# guide = guide_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
# guide = bayes_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, k_dim=1, device=None)

# Setup the variational inference
svi = SVI(bayes_irt_dynamic, guide, optim, loss=Trace_ELBO())
# svi.step(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)

logger.info("Run the variational inference")
pyro.clear_param_store()
# pyro.get_param_store().setdefault("AutoNormal.locs.sigma", init_constrained_value=1.0 * torch.ones(n_legs, k_dim), constraint=constraints.positive)
# pyro.get_param_store().setdefault("AutoNormal.locs.theta", init_constrained_value=custom_init_values)
# pyro.get_param_store().setdefault("AutoNormal.locs.v", init_constrained_value=torch.zeros(n_legs, max_time))

min_loss = float('inf')  # initialize to infinity
patience = 0
for j in tqdm(range(5000)):
    loss = svi.step(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=2, device=None)
    # define optimizer and loss function
    # optimizer = torch.optim.Adam(my_parameters, {"lr": 0.1})
    # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    # compute loss
    # loss = loss_fn(bayes_irt_dynamic, guide, legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
    # loss.backward()
    # take a step and zero the parameter gradients
    # optimizer.step()
    # optimizer.zero_grad()
    if j % 100 == 0:
        logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
        min_loss = min(loss, min_loss)
        if (loss > min_loss):
            if patience >= 4:
                break
            else:
                patience += 1
        else:
            patience = 0





# sparse_votes = torch.sparse.FloatTensor(torch.stack([legs, votes]), responses)
# u, s, v = torch.svd_lowrank(sparse_votes, q=2)
# u.shape
# v.shape
# s.shape

# ideal_point

# def random_walk(i, T):
#     x_0 = pyro.sample(f'x_0_{i}', dist.Normal(0, 1)).unsqueeze(-1)
#     sigma = pyro.sample(f'sigma_{i}', dist.LogNormal(0, 1)).unsqueeze(-1)
#     v = pyro.sample(f'v_{i}', dist.Normal(0, 1).expand([T]).to_event(1))
#     x = pyro.deterministic(f"x_{i}", x_0 + sigma * v.cumsum(dim=-1))
#     return x

# def model_vague(self, models, items, obs):
#     with pyro.plate('thetas', self.num_models, device=self.device):
#         ability = pyro.sample('theta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))
#
#     with pyro.plate('bs', self.num_items, device=self.device):
#         diff = pyro.sample('b', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e3, device=self.device)))
#
#     with pyro.plate('observe_data', obs.size(0), device=self.device):
#         pyro.sample("obs", dist.Bernoulli(logits=ability[models] - diff[items]), obs=obs)
#
# def guide_vague(self, models, items, obs):
#     # register learnable params in the param store
#     m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_models, device=self.device))
#     s_theta_param = pyro.param("scale_ability", torch.ones(self.num_models, device=self.device),
#                                constraint=constraints.positive)
#     m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
#     s_b_param = pyro.param("scale_diff", torch.empty(self.num_items, device=self.device).fill_(1.e3),
#                            constraint=constraints.positive)
#
#     # guide distributions
#     with pyro.plate('thetas', self.num_models, device=self.device):
#         dist_theta = dist.Normal(m_theta_param, s_theta_param)
#         pyro.sample('theta', dist_theta)
#     with pyro.plate('bs', self.num_items, device=self.device):
#         dist_b = dist.Normal(m_b_param, s_b_param)
#         pyro.sample('b', dist_b)


# def is_prng_key(key):
#     try:
#         return key.shape == (2,) and key.dtype == np.uint32
#     except AttributeError:
#         return False
#
#
# def validate_sample(log_prob_fn):
#     def wrapper(self, *args, **kwargs):
#         log_prob = log_prob_fn(self, *args, *kwargs)
#         if self._validate_args:
#             value = kwargs['value'] if 'value' in kwargs else args[0]
#             mask = self._validate_sample(value)
#             log_prob = torch.where(mask, log_prob, -torch.inf)
#         return log_prob
#
#     return wrapper
#
#
# class GaussianRandomWalk(torch.distributions.Distribution):
#     arg_constraints = {'scale': constraints.positive}
#     support = constraints.real_vector
#     reparametrized_params = ['scale']
#
#     def __init__(self, scale=1., num_steps=1, validate_args=None):
#         assert isinstance(num_steps, int) and num_steps > 0, \
#             "`num_steps` argument should be an positive integer."
#         self.scale = scale
#         self.num_steps = num_steps
#         batch_shape, event_shape = torch.shape(scale), (num_steps,)
#         super(GaussianRandomWalk, self).__init__(batch_shape, event_shape, validate_args=validate_args)
#
#     def sample(self, key, sample_shape=()):
#         assert is_prng_key(key)
#         shape = sample_shape + self.batch_shape + self.event_shape
#         walks = torch.random.normal(key, shape=shape)
#         return torch.cumsum(walks, axis=-1) * torch.expand_dims(self.scale, axis=-1)
#
#     @validate_sample
#     def log_prob(self, value):
#         init_prob = torch.distributions.Normal(0., self.scale).log_prob(value[..., 0])
#         scale = torch.expand_dims(self.scale, -1)
#         step_probs = torch.distributions.Normal(value[..., :-1], scale).log_prob(value[..., 1:])
#         return init_prob + torch.sum(step_probs, axis=-1)
#
#     @property
#     def mean(self):
#         return torch.zeros(self.batch_shape + self.event_shape)
#
#     @property
#     def variance(self):
#         return torch.broadcast_to(torch.expand_dims(self.scale, -1) ** 2 * torch.arange(1, self.num_steps + 1),
#                                   self.batch_shape + self.event_shape)
#
#     def tree_flatten(self):
#         return (self.scale,), self.num_steps
#
#     @classmethod
#     def tree_unflatten(cls, aux_data, params):
#         return cls(*params, num_steps=aux_data)
#
#
# def model(data):
#     T = len(data)
#     sigma = pyro.sample('sigma', dist.Exponential(50.))
#     nu = pyro.sample('nu', dist.Exponential(.1))
#     shifts = pyro.sample("h_t", dist.Normal(loc=torch.zeros(T), scale=torch.ones(T) * sigma))
#     h_t = torch.cumsum(shifts, dim=0)
#     y = pyro.sample("returns", dist.StudentT(df=nu, loc=torch.zeros(1), scale=(torch.exp(2*h_t) + 1e-12)), obs=torch.tensor(data))
#     return y
#
#
# def model_vec(data_vec):
#     N = len(data_vec)
#     rv_sigma = pyro.sample("rv_sigma", dist.Exponential(torch.tensor(50.)))
#     rv_nu = pyro.sample("rv_nu", dist.Exponential(torch.tensor(0.1)))
#
#     # random walks model corresponds to scale_tril of tril of all one matrix
#     rv_s = pyro.sample("rv_s", dist.MultivariateNormal(loc=torch.zeros(N)*0.0, scale_tril=torch.ones((N,N)).tril()))
#     pyro.sample("obs",
#                 dist.StudentT(rv_nu, loc=torch.tensor(0.), scale=rv_s.exp()).independent(),
#                 obs=data_vec)



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

class TwoParamLog:
    """2PL IRT model"""
    def __init__(self, priors, device, num_items, num_models, verbose=False):
        if priors not in ['vague', 'hierarchical']:
            raise ValueError("Options for priors are vague and hierarchical")
        if device not in ['cpu', 'gpu']:
            raise ValueError("Options for device are cpu and gpu")
        if num_items <= 0:
            raise ValueError("Number of items must be greater than 0")
        if num_models <= 0:
            raise ValueError("Number of subjects must be greater than 0")
        self.priors = priors
        self.device = device
        self.num_items = num_items
        self.num_models = num_models
        self.verbose = verbose


    def model_vague(self, models, items, obs):
        """Initialize a 2PL model with vague priors"""
        with pyro.plate('thetas', self.num_models, device=self.device):
            ability = pyro.sample('theta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device)))

        with pyro.plate('bs', self.num_items, device=self.device):
            diff = pyro.sample('b', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(0.1, device=self.device)))
            slope = pyro.sample('a', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(0.1, device=self.device)))

        with pyro.plate('observe_data', obs.size(0), device=self.device):
            pyro.sample("obs", dist.Bernoulli(logits=(slope[items]* (ability[models] - diff[items]))), obs=obs)


    def guide_vague(self, models, items, obs):
        """Initialize a 2PL guide with vague priors"""
        # register learnable params in the param store
        m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_models, device=self.device))
        s_theta_param = pyro.param("scale_ability", torch.ones(self.num_models, device=self.device),
                            constraint=constraints.positive)
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param("scale_diff", torch.empty(self.num_items, device=self.device).fill_(1.e1),
                                constraint=constraints.positive)
        m_a_param = pyro.param("loc_slope", torch.ones(self.num_items, device=self.device),
                                constraint=constraints.positive)
        s_a_param = pyro.param("scale_slope", torch.empty(self.num_items, device=self.device).fill_(1.e-6),
                                constraint=constraints.positive)

        # guide distributions
        with pyro.plate('thetas', self.num_models, device=self.device):
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample('theta', dist_theta)
        with pyro.plate('bs', self.num_items, device=self.device):
            dist_b = dist.Normal(m_b_param, s_b_param)
            pyro.sample('b', dist_b)

            dist_a = dist.Normal(m_a_param, s_a_param)
            pyro.sample('a', dist_a)


    def model_hierarchical(self, models, items, obs):
        """Initialize a 2PL model with hierarchical priors"""
        mu_b = pyro.sample('mu_b', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e6, device=self.device)))
        u_b = pyro.sample('u_b', dist.Gamma(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))
        mu_theta = pyro.sample('mu_theta', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e6, device=self.device)))
        u_theta = pyro.sample('u_theta', dist.Gamma(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))
        mu_a = pyro.sample('mu_a', dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1.e6, device=self.device)))
        u_a = pyro.sample('u_a', dist.Gamma(torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)))
        with pyro.plate('thetas', self.num_models, device=self.device):
            ability = pyro.sample('theta', dist.Normal(mu_theta, 1. / u_theta))
        with pyro.plate('bs', self.num_items, device=self.device):
            diff = pyro.sample('b', dist.Normal(mu_b, 1. / u_b))
            slope = pyro.sample('a', dist.Normal(mu_a, 1. / u_a))
        with pyro.plate('observe_data', obs.size(0)):
            pyro.sample("obs", dist.Bernoulli(logits=slope[items] * (ability[models] - diff[items])), obs=obs)


    def guide_hierarchical(self, models, items, obs):
        """Initialize a 2PL guide with hierarchical priors"""
        loc_mu_b_param = pyro.param('loc_mu_b', torch.tensor(0., device=self.device))
        scale_mu_b_param = pyro.param('scale_mu_b', torch.tensor(1.e1, device=self.device),
                                constraint=constraints.positive)
        loc_mu_theta_param = pyro.param('loc_mu_theta', torch.tensor(0., device=self.device))
        scale_mu_theta_param = pyro.param('scale_mu_theta', torch.tensor(1.e1, device=self.device),
                            constraint=constraints.positive)
        loc_mu_a_param = pyro.param('loc_mu_a', torch.tensor(0., device=self.device))
        scale_mu_a_param = pyro.param('scale_mu_a', torch.tensor(1.e1, device=self.device),
                            constraint=constraints.positive)
        alpha_b_param = pyro.param('alpha_b', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        beta_b_param = pyro.param('beta_b', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        alpha_theta_param = pyro.param('alpha_theta', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        beta_theta_param = pyro.param('beta_theta', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        alpha_a_param = pyro.param('alpha_a', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        beta_a_param = pyro.param('beta_a', torch.tensor(1., device=self.device),
                        constraint=constraints.positive)
        m_theta_param = pyro.param('loc_ability', torch.zeros(self.num_models, device=self.device))
        s_theta_param = pyro.param('scale_ability', torch.ones(self.num_models, device=self.device),
                            constraint=constraints.positive)
        m_b_param = pyro.param('loc_diff', torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param('scale_diff', torch.ones(self.num_items, device=self.device),
                                constraint=constraints.positive)
        m_a_param = pyro.param('loc_slope', torch.zeros(self.num_items, device=self.device))
        s_a_param = pyro.param('scale_slope', torch.ones(self.num_items, device=self.device),
                                constraint=constraints.positive)

        # sample statements
        pyro.sample('mu_b', dist.Normal(loc_mu_b_param, scale_mu_b_param))
        pyro.sample('u_b', dist.Gamma(alpha_b_param, beta_b_param))
        pyro.sample('mu_theta', dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        pyro.sample('u_theta', dist.Gamma(alpha_theta_param, beta_theta_param))
        pyro.sample('mu_a', dist.Normal(loc_mu_a_param, scale_mu_a_param))
        pyro.sample('u_a', dist.Gamma(alpha_a_param, beta_a_param))


        with pyro.plate('thetas', self.num_models, device=self.device):
            pyro.sample('theta', dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate('bs', self.num_items, device=self.device):
            pyro.sample('b', dist.Normal(m_b_param, s_b_param))
            pyro.sample('a', dist.Normal(m_a_param, s_a_param))


    def fit(self, models, items, responses, num_epochs):
        """Fit the IRT model with variational inference"""
        optim = Adam({'lr': 0.1})
        if self.priors == 'vague':
            svi = SVI(self.model_vague, self.guide_vague, optim, loss=Trace_ELBO())
        else:
            svi = SVI(self.model_hierarchical, self.guide_hierarchical, optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))
        values = ['loc_diff', 'scale_diff', 'loc_ability', 'scale_ability']


    def fit_MCMC(self, models, items, responses, num_epochs):
        """Fit the IRT model with MCMC"""
        sites = ['theta', 'b']
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100) \
            .run(models, items, responses)
        theta_sum = self.summary(hmc_posterior, ['theta']).items()
        b_sum = self.summary(hmc_posterior, ['b']).items()
        print(theta_sum)
        print(b_sum)


    def summary(self, traces, sites):
        """Aggregate marginals for MCM"""
        marginal = EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1) \
                [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
        return site_stats


model = TwoParamLog(priors="vague", device="cpu", num_items=n_votes, num_models=n_legs, verbose=True)
model.fit(legs, votes, responses, 500)





# test_model(model, autoguides.AutoNormal(bayes_irt_dynamic), Trace_ELBO())


logger.info("Test a model with covariates")
logger.info("Setup the Bayesian Model")
# Choose the optimizer used by the variational algorithm
optim = Adam({'lr': 0.1, 'weight_decay': 0.00001})

# Define the guide, intialize to the values returned from process data
# guide = autoguides.AutoNormal(bayes_irt_dynamic)
# guide = guide_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
guide = bayes_irt_dynamic(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, k_dim=1, device=None)

# Setup the variational inference
svi = SVI(bayes_irt_dynamic, guide, optim, loss=Trace_ELBO())
# svi.step(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)

logger.info("Run the variational inference")
pyro.clear_param_store()
# pyro.get_param_store().setdefault("AutoNormal.locs.sigma", init_constrained_value=1.0 * torch.ones(n_legs, k_dim), constraint=constraints.positive)
# pyro.get_param_store().setdefault("AutoNormal.locs.theta", init_constrained_value=custom_init_values)
# pyro.get_param_store().setdefault("AutoNormal.locs.v", init_constrained_value=torch.zeros(n_legs, max_time))

min_loss = float('inf')  # initialize to infinity
patience = 0
for j in tqdm(range(5000)):
    loss = svi.step(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
    # define optimizer and loss function
    # optimizer = torch.optim.Adam(my_parameters, {"lr": 0.1})
    # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    # compute loss
    # loss = loss_fn(bayes_irt_dynamic, guide, legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=responses, k_dim=1, device=None)
    # loss.backward()
    # take a step and zero the parameter gradients
    # optimizer.step()
    # optimizer.zero_grad()
    if j % 100 == 0:
        logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
        min_loss = min(loss, min_loss)
        if (loss > min_loss):
            if patience >= 4:
                break
            else:
                patience += 1
        else:
            patience = 0

indices = (torch.arange(start=0, end=max_time + 1).repeat(n_legs, 1))
mask = indices.le(sessions_served.unsqueeze(-1))

init_ideal_point = pyro.param("AutoNormal.locs.theta")
atten = pyro.param("AutoNormal.locs.sigma")
atten.shape
disturb = pyro.param("AutoNormal.locs.v")
disturb.shape
disturb.max(axis=1) == disturb.max()
# disturb
disturb.shape

pyro.param("AutoNormal.locs.sigma")
pyro.param("AutoNormal.scales.sigma")

pyro.param("AutoNormal.locs.v") * mask[:, 1:]
pyro.param("AutoNormal.scales.v") * mask[:, 1:]


init_ideal_point.shape
later_ideal_point = init_ideal_point + (atten * disturb).squeeze().cumsum(dim=-1)
later_ideal_point.shape
ideal_point = torch.hstack([init_ideal_point, later_ideal_point])
ideal_point.shape

ideal_point * mask

# ideal_point
# temp_ideal = pd.DataFrame((ideal_point * mask).detach().numpy())
# temp_ideal[temp_ideal[6] != 0].sort_values(6)
# temp_ideal[(temp_ideal[0].abs() > temp_ideal[6].abs()) & (temp_ideal[6] != 0)].sort_values(6)



n_samples = 1000

# This is the "normal" case, our models are large enough we exhaust memory, hence the iterative approach
# posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, guide=guide, num_samples=n_samples, return_sites=["obs"])
# preds = posterior_predictive(legs, votes, covariates=covariates, device=device)["obs"].mean(axis=0).squeeze()
# preds_test = posterior_predictive(legs_test, votes_test, covariates=covariates_test, device=device)["obs"].mean(axis=0).squeeze()

posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_dynamic, guide=guide, num_samples=1, return_sites=["obs", "ideal_point"])

preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
ideal_point_list = []
for _ in tqdm(range(n_samples)):
    in_sample_predictive = posterior_predictive(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=None, k_dim=1, device=None)
    preds += in_sample_predictive["obs"].squeeze()
    ideal_point_list += [in_sample_predictive["ideal_point"].squeeze()]
    preds_test += posterior_predictive(legs_test, n_legs, votes_test, n_votes, ind_time_tensor_test, max_time, y=None, k_dim=1, device=None)["obs"].squeeze()
preds = preds / n_samples
preds_test = preds_test / n_samples

ideal_point_list[10]
torch.stack(ideal_point_list, dim=-1).mean(axis=2) * mask
ideal_point * mask
torch.stack(ideal_point_list, dim=-1).std(axis=2) * mask

pyro.get_param_store().keys()
pyro.infer.predictive.Predictive(bayes_irt_dynamic, guide=guide, num_samples=1)(legs, n_legs, votes, n_votes, ind_time_tensor, max_time, y=None, k_dim=1, device=None)["ideal_point"]
pyro.param("AutoNormal.locs.v")

# Define metrics
criterion = torch.nn.BCELoss(reduction="mean")
log_like = torch.nn.BCELoss(reduction="sum")
k = 0
v_indices = (torch.arange(start=0, end=max_time).repeat(n_legs, 1))
v_mask = v_indices.le(sessions_served.unsqueeze(-1))
for param_name, param_value in pyro.get_param_store().items():
    print(param_name)
    if "locs" in param_name:
        if ".v" in param_name:
            # Use only relevant entries for v
            k += (param_value * v_mask != 0).sum().item()
        else:
            k += np.array(param_value.shape).prod()

pyro.get_param_store().keys()
pyro.param("AutoNormal.locs.v")

# Calculate log loss and accuracy scores, in sample
train_metrics = {"bce": criterion(preds, responses).item(),
                 "log_like": log_like(preds, responses).item(),
                 "accuracy": (1.0 * (responses == preds.round())).mean().item(),
                }
logger.info(f'Train loss (VB): {train_metrics["bce"]}')
logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

# Calculate log loss and accuracy scores, out of sample
test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                "log_like": log_like(preds_test, responses_test).item(),
                "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(),
                }
logger.info(f'Test loss (VB): {test_metrics["bce"]}')
logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

metrics = {**train_metrics, **test_metrics}

metrics["train_n"] = responses.shape[0]
metrics["train_k"] = k
metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

metrics["test_n"] = responses_test.shape[0]
metrics["test_k"] = k
metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])

final_metrics = {**data_params, **metrics}
final_metrics





# init_ideal_point.shape
# later_ideal_point = init_ideal_point + (atten * disturb).squeeze().cumsum(dim=-1)
# later_ideal_point.shape

init_ideal_point = pyro.param("AutoNormal.locs.theta")
atten = pyro.param("AutoNormal.locs.sigma")
disturb = pyro.param("AutoNormal.locs.v")
ideal_point = torch.hstack([init_ideal_point, init_ideal_point + (atten * disturb).cumsum(dim=-1)])
ideal_point.shape

indices = (torch.arange(start=0, end=max_time + 1).repeat(len(ideal_point), 1))
mask = indices.le(sessions_served.unsqueeze(-1))

ideal_point
temp_ideal = pd.DataFrame((ideal_point * mask).detach().numpy())
temp_ideal[temp_ideal[6] != 0].sort_values(6)
temp_ideal[(temp_ideal[0].abs() > temp_ideal[6].abs()) & (temp_ideal[6] != 0)].sort_values(6)

leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

leg_data[leg_data["leg_id"] == vote_data["leg_crosswalk"][682]]

leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}

leg_data[leg_data["bioname"].str.contains("FLAKE")]
leg_data[leg_data["bioname"].str.contains("McCAIN")]
leg_crosswalk_rev[20100]
temp_ideal.loc[603]
leg_crosswalk_rev[15039]
temp_ideal.loc[1073]

temp_ideal.sort_values(0)

leg_data[leg_data["leg_id"] == vote_data["leg_crosswalk"][252]]

temp_ideal[0].hist()






# Set some constants
n_legs = vote_data["J"]
n_votes = vote_data["M"]
if covariates_list:
    n_covar = covariates.shape[1]

# Make train and test data bundles
core_data = [legs, n_legs, votes, n_votes]
core_data_test = [legs_test, n_legs, votes_test, n_votes]
aux_data = {}
aux_data_test = {}
if covariates_list:
    aux_data["k_dim"] = k_dim
    aux_data["covariates"] = covariates
    aux_data["n_covar"] = n_covar

    aux_data_test["k_dim"] = k_dim
    aux_data_test["covariates"] = covariates_test
    aux_data_test["n_covar"] = n_covar
if k_time > 0:
    aux_data["k_time"] = k_time
    aux_data["time_passed"] = time_tensor

    aux_data_test["k_time"] = k_time
    aux_data_test["time_passed"] = time_tensor_test

# Setup the optimizer
optim = Adam({'lr': 0.1, 'weight_decay': 0.1})

# Define the guide
guide = autoguides.AutoNormal(bayes_irt_full)
# guide = autoguides.AutoNormal(bayes_irt_basic, init_loc_fn=init_to_value(values={'theta': custom_init_values}))

# Setup the variational inference
svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())

logger.info("Fit the model using variational bayes")
# Run variational inference
pyro.clear_param_store()
min_loss = float('inf') # initialize to infinity
patience = 0
for j in tqdm(range(5000)):
    loss = svi.step(*core_data, y=responses, device=device, **aux_data)
    if j % 100 == 0:
        logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
        min_loss = min(loss, min_loss)
        if (loss > min_loss):
            if patience >= 4:
                break
            else:
                patience += 1
        else:
            patience = 0

n_samples = 1000

# This is the "normal" case, our models are large enough we exhaust memory, hence the iterative approach
# posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, guide=guide, num_samples=n_samples, return_sites=["obs"])
# preds = posterior_predictive(legs, votes, covariates=covariates, device=device)["obs"].mean(axis=0).squeeze()
# preds_test = posterior_predictive(legs_test, votes_test, covariates=covariates_test, device=device)["obs"].mean(axis=0).squeeze()

posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, guide=guide, num_samples=1, return_sites=["obs"])

preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
for _ in tqdm(range(n_samples)):
    preds += posterior_predictive(*core_data, device=device, **aux_data)["obs"].squeeze()
    preds_test += posterior_predictive(*core_data_test, device=device, **aux_data_test)["obs"].squeeze()
preds = preds / n_samples
preds_test = preds_test / n_samples

# Define metrics
criterion = torch.nn.BCELoss(reduction="mean")
log_like = torch.nn.BCELoss(reduction="sum")
k = 0
for param_name, param_value in pyro.get_param_store().items():
    if "locs" in param_name:
        k += np.array(param_value.shape).prod()

# Calculate log loss and accuracy scores, in sample
train_metrics = {"bce": criterion(preds, responses).item(),
                 "log_like": log_like(preds, responses).item(),
                 "accuracy": (1.0 * (responses == preds.round())).mean().item(),
                }
logger.info(f'Train loss (VB): {train_metrics["bce"]}')
logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

# Calculate log loss and accuracy scores, out of sample
test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                "log_like": log_like(preds_test, responses_test).item(),
                "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(),
                }
logger.info(f'Test loss (VB): {test_metrics["bce"]}')
logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

metrics = {**train_metrics, **test_metrics}

metrics["train_n"] = responses.shape[0]
metrics["train_k"] = k
metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

metrics["test_n"] = responses_test.shape[0]
metrics["test_k"] = k
metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])

final_metrics = {**data_params, **metrics}
final_metrics



import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

# enable validation (e.g. validate parameters of distributions)
# assert pyro.__version__.startswith('1.5.2')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
