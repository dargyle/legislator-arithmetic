import pyro
import pyro.distributions as dist
import torch

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides
from torch.distributions import constraints

import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item

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

indices = (torch.arange(start=1, end=max_time + 1, device=device).repeat(n_legs, 1))
disturb_mask = indices.le(sessions_served.unsqueeze(-1)).unsqueeze(1)

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
    def __init__(self, n_legs, n_votes, max_time, k_dim=1, guide_type="auto", custom_init_values=None, disturb_mask=None, device=None):
        super().__init__()
        self.n_legs = n_legs
        self.n_votes = n_votes
        self.max_time = max_time
        self.k_dim = k_dim
        self.guide_type = guide_type
        self.custom_init_values = custom_init_values
        self.device = device
        self.disturb_mask = disturb_mask

    def model(self, legs, votes, ind_time_tensor, obs=None):
        # Declare context managers, implies conditional independence across the specified dimension
        # 3 Dimensions: legislator/vote/observation, ideology dimension, time dimension
        leg_plate = pyro.plate('legs', self.n_legs, dim=-3, device=self.device)
        vote_plate = pyro.plate('votes', self.n_votes, dim=-3, device=self.device)
        dim_plate = pyro.plate('dim', self.k_dim, dim=-2, device=self.device)
        time_plate = pyro.plate('time', self.max_time, dim=-1, device=self.device)

        with leg_plate, dim_plate:
            init_ideal_point = pyro.sample('theta', dist.Normal(0, 1))
            # atten = pyro.sample('tau', dist.Gamma(2, 0.1))
            atten = pyro.sample('tau', dist.HalfNormal(1.0))

        if self.disturb_mask is not None:
            with leg_plate, dim_plate, time_plate:
                with poutine.mask(mask=self.disturb_mask):
                    disturb = pyro.sample('v', dist.Normal(0, 1))
                    combo_disturb = (atten * disturb).cumsum(-1)
        else:
            with leg_plate, dim_plate, time_plate:
                disturb = pyro.sample('v', dist.Normal(0, 1))
                combo_disturb = (atten * disturb).cumsum(-1)

        with vote_plate, dim_plate:
            polarity = pyro.sample('beta', dist.Normal(0, 5))

        with vote_plate:
            popularity = pyro.sample('alpha', dist.Normal(0, 5))

        # print(init_ideal_point.shape)
        # print(atten.shape)
        # print(disturb.shape)
        # print((atten * disturb).cumsum(dim=2).shape)
        ideal_point = pyro.deterministic("ideal_point", torch.cat([init_ideal_point, init_ideal_point + combo_disturb], dim=-1))

        batch_ideal = ideal_point[legs, :, ind_time_tensor]
        polarity_batch = polarity[votes]
        popularity_batch = popularity[votes]
        # print(batch_ideal.device)
        # print(polarity_batch.shape)
        # print(popularity_batch.shape)

        # Combine parameters
        # print(torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1).shape)
        # print(popularity_batch.squeeze().shape)
        logit = torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1) + popularity_batch.squeeze()
        # print(logit.shape)

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
        loc_theta = pyro.param("loc_init_ideal", lambda: torch.zeros(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1))
        scale_theta = pyro.param("scale_init_ideal", lambda: torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1),
                                 constraint=constraints.positive)
        # loc_tau = pyro.param("loc_atten", lambda: torch.zeros(self.n_legs, self.k_dim, device=device).unsqueeze(-1),
        #                        constraint=constraints.positive)
        scale_tau = pyro.param("scale_atten", lambda: torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1),
                                 constraint=constraints.positive)
        loc_v = pyro.param("loc_disturb", lambda: torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
        scale_v = pyro.param("scale_disturb", lambda: torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device),
                             constraint=constraints.positive)
        loc_beta = pyro.param("loc_polarity", lambda: torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
        scale_beta = pyro.param("scale_polarity", lambda: 5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1),
                                constraint=constraints.positive)
        loc_alpha = pyro.param("loc_popularity", lambda: torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
        scale_alpha = pyro.param("scale_popularity", lambda: 5.0 * torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1),
                                 constraint=constraints.positive)

        leg_plate = pyro.plate('legs', self.n_legs, dim=-3, device=self.device)
        vote_plate = pyro.plate('votes', self.n_votes, dim=-3, device=self.device)
        dim_plate = pyro.plate('dim', self.k_dim, dim=-2, device=self.device)
        time_plate = pyro.plate('time', self.max_time, dim=-1, device=self.device)

        with leg_plate, dim_plate:
            pyro.sample('theta', dist.Normal(loc_theta, scale_theta))
            pyro.sample('tau', dist.HalfNormal(scale_tau))

        if self.disturb_mask is not None:
            with leg_plate, dim_plate, time_plate:
                with poutine.mask(mask=self.disturb_mask):
                    pyro.sample('v', dist.Normal(loc_v, scale_v))
        else:
            with leg_plate, dim_plate, time_plate:
                pyro.sample('v', dist.Normal(loc_v, scale_v))

        with vote_plate, dim_plate:
            pyro.sample('beta', dist.Normal(loc_beta, scale_beta))

        with vote_plate:
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
            pyro.get_param_store().get_param("AutoNormal.locs.theta", init_tensor=self.custom_init_values)
            pyro.get_param_store().get_param("AutoNormal.scales.theta", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.tau", init_tensor=0.1 * torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.scales.tau", init_tensor=0.1 * torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.v", init_tensor=torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
            pyro.get_param_store().get_param("AutoNormal.scales.v", init_tensor=torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device), constraint=constraints.interval(1e-7, 1e7))
            pyro.get_param_store().get_param("AutoNormal.locs.beta", init_tensor=torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
            pyro.get_param_store().get_param("AutoNormal.scales.beta", init_tensor=torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.alpha", init_tensor=torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
            pyro.get_param_store().get_param("AutoNormal.scales.alpha", init_tensor=torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1), constraint=constraints.positive)
        elif self.guide_type == "basic":
            pyro.get_param_store().get_param("loc_init_ideal", init_tensor=self.custom_init_values)
            # pyro.get_param_store().get_param("scale_init_ideal", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_atten", init_tensor=0.1 * torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("scale_atten", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_disturb", init_tensor=torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
            # pyro.get_param_store().get_param("scale_disturb", init_tensor=torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_polarity", init_tensor=torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
            # pyro.get_param_store().get_param("scale_polarity", init_tensor=5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_popularity", init_tensor=torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
            # pyro.get_param_store().get_param("scale_popularity", init_tensor=5.0 * torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1), constraint=constraints.positive)

        logger.info("Run the variational inference")
        # Setup the variational inference
        optim = Adam({'lr': 0.05})
        svi = custom_SVI(self.model, guide, optim, loss=Trace_ELBO(), l2_penalty=0.0)
        self.guide_used = guide

        min_loss = float('inf')  # initialize to infinity
        patience = 0
        for j in tqdm(range(15000), ncols=100, mininterval=1):
            loss = svi.step(legs, votes, ind_time_tensor, responses)
            # loss = svi.penalized_step(legs, votes, ind_time_tensor, responses)

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
test = bayes_dynamic(n_legs, n_votes, max_time, k_dim=k_dim, guide_type="basic", custom_init_values=custom_init_values.unsqueeze(-1), disturb_mask=disturb_mask, device=device)
test.fit(legs, votes, ind_time_tensor, responses)

preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
ideal_point_list = []
n_samples = 1000
for _ in tqdm(range(n_samples)):
    guide_trace = pyro.poutine.trace(test.guide_used).get_trace(legs, votes, ind_time_tensor)
    posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(test.model, guide_trace)).get_trace(legs, votes, ind_time_tensor)
    preds += posterior_predictive.nodes['obs']['value'].squeeze()
    # print(posterior_predictive.nodes["v"]['value'].squeeze())
    # ideal_point_list += [posterior_predictive.nodes["ideal_point"]['value'].squeeze()]

    guide_trace = pyro.poutine.trace(test.guide_used).get_trace(legs_test, votes_test, ind_time_tensor_test)
    posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(test.model, guide_trace)).get_trace(legs_test, votes_test, ind_time_tensor_test)
    preds_test += posterior_predictive.nodes['obs']['value'].squeeze()
preds = preds / n_samples
preds_test = preds_test / n_samples


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


logger.info("Now fit the model using sampling")
# Initialize the MCMC to our estimates from the variational model
# Speeds things up and requires less burn-in
if test.guide_type == "auto":
    if covariates_list:
        init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                            "beta": pyro.param("AutoNormal.locs.beta").data,
                                            "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                            "tau": pyro.param("AutoNormal.locs.tau").data,
                                            "v": pyro.param("AutoNormal.locs.v").data * disturb_mask,
                                            "coef": pyro.param("AutoNormal.locs.coef").data,
                                            })
    else:
        init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                            "beta": pyro.param("AutoNormal.locs.beta").data,
                                            "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                            "tau": pyro.param("AutoNormal.locs.tau").data,
                                            "v": pyro.param("AutoNormal.locs.v").data * disturb_mask,
                                            })
elif test.guide_type == "basic":
    if covariates_list:
        init_values = init_to_value(values={"theta": pyro.param("loc_init_ideal").data,
                                            "beta": pyro.param("loc_polarity").data,
                                            "alpha": pyro.param("loc_popularity").data,
                                            "tau": pyro.param("scale_atten").data,
                                            "v": pyro.param("loc_disturb").data * disturb_mask,
                                            "coef": pyro.param("AutoNormal.locs.coef").data,
                                            })
    else:
        init_values = init_to_value(values={"theta": pyro.param("loc_init_ideal").data,
                                            "beta": pyro.param("loc_polarity").data,
                                            "alpha": pyro.param("loc_popularity").data,
                                            "tau": pyro.param("scale_atten").data,
                                            "v": pyro.param("loc_disturb").data * disturb_mask,
                                            })

# Set up sampling alrgorithm
# kernel = HMC(test.model, adapt_step_size=True, target_accept_prob=0.85, step_size=1e-4, num_steps=25, jit_compile=False, ignore_jit_warnings=True, init_strategy=init_values)
kernel = NUTS(test.model, step_size=1e-4, adapt_step_size=True, max_tree_depth=7, target_accept_prob=0.85, jit_compile=False, ignore_jit_warnings=True, use_multinomial_sampling=False)
# For real inference should probably increase the number of samples, but this is part is slow and enough to test
hmc_posterior = MCMC(kernel, num_samples=250, warmup_steps=50, disable_validation=False)
# Run the model
hmc_posterior.run(legs, votes, ind_time_tensor, obs=responses)

# # Save the posterior
# hmc_posterior.sampler = None
# hmc_posterior.kernel.potential_fn = None
# with open(US_PATH + f'bayes/params_mcmc_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl', 'wb') as f:
#     pickle.dump(hmc_posterior, f)
# # with open(US_PATH + f'bayes/params_mcmc_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl', 'rb') as f:
# #     hmc_posterior = pickle.load(f)





# Declare context managers, implies conditional independence across the specified dimension
# 3 Dimensions: legislator/vote/observation, ideology dimension, time dimension
leg_plate = pyro.plate('legs', n_legs, dim=-3, device=device)
vote_plate = pyro.plate('votes', n_votes, dim=-3, device=device)
dim_plate = pyro.plate('dim', k_dim, dim=-2, device=device)
time_plate = pyro.plate('time', max_time, dim=-1, device=device)

with leg_plate, dim_plate:
    init_ideal_point = pyro.sample('theta', dist.Normal(0, 1))
    atten = pyro.sample('tau', dist.Gamma(1, 1))

with leg_plate, dim_plate, time_plate:
    with poutine.mask(mask=disturb_mask):
        disturb = pyro.sample('v', dist.Normal(0, 1))

with vote_plate, dim_plate:
    polarity = pyro.sample('beta', dist.Normal(0, 5))

with vote_plate:
    popularity = pyro.sample('alpha', dist.Normal(0, 5))

# temp_ideal = init_ideal_point.unsqueeze(-1)
# print(temp_ideal.shape)
# print(atten.shape)
# print(disturb.shape)
# all_temp_ideal = init_ideal_point + (atten * disturb).cumsum(dim=2)
# all_temp_ideal.shape
# print(all_temp_ideal.shape)

ideal_point = pyro.deterministic("ideal_point", torch.cat([init_ideal_point, init_ideal_point + (atten * disturb).cumsum(dim=2)], dim=2))
ideal_point.shape

batch_ideal = ideal_point[legs, :, ind_time_tensor]
polarity_batch = polarity[votes]
popularity_batch = popularity[votes]

batch_ideal.shape
polarity_batch.shape
popularity_batch.shape

torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1).shape
popularity_batch.squeeze().shape

# Combine parameters
logit = torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1) + popularity_batch.squeeze()

obs = responses
obs.shape

if obs is not None:
    # If training outcomes are provided, run the sampling statement for the given data
    with pyro.plate('observe_data', obs.size(0), device=device):
        pyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)
else:
    # If no training outcomes are provided, return the samples from the predicted distributions
    with pyro.plate('observe_data', legs.size(0), device=self.device):
        obs = pyro.sample("obs", dist.Bernoulli(logits=logit))
    return obs




init_ideal_point = pyro.sample('theta', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))

with pyro.plate('sigmas', n_legs, dim=-2, device=device):
    # atten = pyro.sample('sigma', dist.Exponential(1.0 * torch.ones(k_dim, device=device)))
    # atten = pyro.sample('sigma', dist.Normal(torch.zeros(k_dim, device=device), torch.ones(k_dim, device=device)))
    atten = pyro.sample('sigma', dist.Gamma(torch.ones(k_dim, device=device), torch.ones(k_dim, device=device)))

with pyro.plate('vs', n_legs, dim=-3, device=device):
    disturb = pyro.sample('v', dist.Normal(torch.zeros(k_dim, max_time, device=device), torch.ones(k_dim, max_time, device=device)))

with pyro.plate('betas', n_votes, dim=-2,  device=device):
    polarity = pyro.sample('beta', dist.Normal(torch.zeros(k_dim, device=device), 5.0 * torch.ones(k_dim, device=device)))

with pyro.plate('alphas', n_votes, dim=-2, device=device):
    popularity = pyro.sample('alpha', dist.Normal(torch.zeros(1, device=device), 5.0 * torch.ones(1, device=device)))



pyro.clear_param_store()
test = bayes_dynamic(n_legs, n_votes, max_time, k_dim=k_dim, guide_type="auto", custom_init_values=custom_init_values, device=device)
test.fit(legs, votes, ind_time_tensor, responses)
pyro.get_param_store().keys()

posterior_predictive = pyro.infer.predictive.Predictive(model=test.model, guide=test.guide_used, num_samples=1, return_sites=["obs", "ideal_point"])
# autoguides.AutoNormal(self.model)

preds = torch.zeros(size=responses.shape, device=device)
preds_test = torch.zeros(size=responses_test.shape, device=device)
ideal_point_list = []
n_samples = 1000
for _ in tqdm(range(n_samples)):
    guide_trace = pyro.poutine.trace(test.guide_used).get_trace(legs, votes, ind_time_tensor)
    posterior_predictive = pyro.poutine.trace(pyro.poutine.replay(test.model, guide_trace)).get_trace(legs, votes, ind_time_tensor)
    preds += posterior_predictive.nodes['obs']['value'].squeeze()
    ideal_point_list += [posterior_predictive.nodes["ideal_point"]['value'].squeeze()]

    guide_trace = pyro.poutine.trace(test.guide_used).get_trace(legs_test, votes_test, ind_time_tensor_test)
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
indices.le(sessions_served.unsqueeze(-1)).shape
mask = indices.le(sessions_served.unsqueeze(-1)).unsqueeze(1)
mask.shape
(mask * posterior_predictive.nodes['ideal_point']['value']).shape

torch.stack(ideal_point_list, dim=-1).mean(axis=-1).shape
(torch.stack(ideal_point_list, dim=-1).mean(axis=-1) * mask).shape
torch.stack(ideal_point_list, dim=-1).std(axis=-1) * mask
temp_ideals = torch.stack(ideal_point_list, dim=-1).mean(axis=-1) * mask
ideal_df_list = []
for i in range(k_dim):
    temp_df = pd.DataFrame(temp_ideals[:, i, :].detach().numpy())
    temp_df.index.name = "leg_id"
    temp_df.columns.name = "time_index"
    temp_df = temp_df.replace({0.0: np.nan}).stack().dropna().to_frame("dim_" + str(i))
    ideal_df_list += [temp_df]
ideal_df = pd.concat(ideal_df_list, axis=1)

leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")

# leg_data[leg_data["leg_id"] == vote_data["leg_crosswalk"][1065]]
# ideal_df.loc[1065, :]

leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}

# leg_data[leg_data["bioname"].str.contains("FLAKE")]
# leg_data[leg_data["bioname"].str.contains("McCAIN")]
# leg_data[leg_data["bioname"].str.contains("GILLIBRAND")]
# leg_data[leg_data["bioname"].str.contains("SANDERS")]
# leg_data[leg_data["bioname"].str.contains("HATCH")]
# leg_data[leg_data["bioname"].str.contains("BIDEN")]

ideal_df.loc[leg_crosswalk_rev[20100]]
ideal_df.loc[leg_crosswalk_rev[15039]]
ideal_df.loc[leg_crosswalk_rev[20735]]
ideal_df.loc[leg_crosswalk_rev[29147]]
ideal_df.loc[leg_crosswalk_rev[14503]]
ideal_df.loc[leg_crosswalk_rev[14101]]


d = pyro.distributions.Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)

type(x)
x.expand_by()


def is_prng_key(key):
    try:
        return key.shape == (2,) and key.dtype == np.uint32
    except AttributeError:
        return False


def validate_sample(log_prob_fn):
    def wrapper(self, *args, **kwargs):
        log_prob = log_prob_fn(self, *args, *kwargs)
        if self._validate_args:
            value = kwargs['value'] if 'value' in kwargs else args[0]
            mask = self._validate_sample(value)
            log_prob = torch.where(mask, log_prob, -torch.inf)
        return log_prob

    return wrapper


from pyro.distributions.distribution import Distribution
from pyro.distributions import Normal
import torch.random as random


class GaussianRandomWalk(dist.TorchDistribution):
    arg_constraints = {'scale': constraints.positive}
    support = constraints.real_vector
    reparametrized_params = ['scale']

    def __init__(self, scale=torch.ones(1), num_steps=1, validate_args=None):
        assert isinstance(num_steps, int) and num_steps > 0, \
            "`num_steps` argument should be an positive integer."
        self.scale = scale
        self.num_steps = num_steps
        # self.batch_shape, self.event_shape = scale.shape, (num_steps,)
        self.validate_args = validate_args
        super(GaussianRandomWalk, self).__init__()

    def sample(self, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = torch.normal(mean=0, std=1, size=shape)
        return torch.cumsum(walks, axis=-1) * self.scale.unsqueeze(-1)

    # @validate_sample
    def log_prob(self, value):
        init_prob = Normal(0., self.scale).log_prob(value[..., 0])
        scale = self.scale.unsqueeze(-1)
        step_probs = Normal(value[..., :-1], scale).log_prob(value[..., 1:])
        return init_prob + torch.sum(step_probs, axis=-1)

    @staticmethod
    def infer_shapes(self):
        return self.batch_shape, self.event_shape

    @property
    def mean(self):
        return torch.zeros(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return torch.broadcast_to(torch.expand(self.scale, -1) ** 2 * torch.arange(1, self.num_steps + 1),
                                  self.batch_shape + self.event_shape)

    def tree_flatten(self):
        return (self.scale,), self.num_steps

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, num_steps=aux_data)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(GaussianRandomWalk, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.num_steps = self.num_steps.expand(batch_shape)
        return super(GaussianRandomWalk, self).expand(batch_shape, new)


class GaussianRandomWalk(dist.TorchDistribution):
    has_rsample = True
    arg_constraints = {'scale': constraints.positive}
    support = constraints.real

    def __init__(self, scale, num_steps=1):
        self.scale = scale
        batch_shape, event_shape = scale.shape, torch.Size([num_steps])
        super(GaussianRandomWalk, self).__init__(batch_shape, event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = self.scale.new_empty(shape).normal_()
        return walks.cumsum(-1) * self.scale.unsqueeze(-1)

    def log_prob(self, x):
        init_prob = dist.Normal(self.scale.new_tensor(0.), self.scale).log_prob(x[..., 0])
        step_probs = dist.Normal(x[..., :-1], self.scale.unsqueeze(-1)).log_prob(x[..., 1:])
        return init_prob + step_probs.sum(-1)


temp_scale = torch.ones(3)
temp_scale.shape
asdf = GaussianRandomWalk(scale=temp_scale, num_steps=5)
x = asdf.sample()
x.shape
init_prob = dist.Normal(asdf.scale.new_tensor(0.), asdf.scale).log_prob(x[..., 0])
step_probs = dist.Normal(loc=x[..., :-1], scale=asdf.scale.unsqueeze(-1)).log_prob(x[..., 1:])

asdf.event_shape
asdf.batch_shape
val = asdf.sample()
asdf.log_prob(val)

zxcv = Normal(torch.tensor([0.0, 0.0]).expand(2, 2), torch.tensor([1.0, 1.0]).expand(2, 2))
val = zxcv.sample()
zxcv.log_prob(val)

leg_plate = pyro.plate('legs', n_legs, dim=-3, device=device)
vote_plate = pyro.plate('votes', n_votes, dim=-3, device=device)
dim_plate = pyro.plate('dim', k_dim, dim=-2, device=device)
# time_plate = pyro.plate('time', max_time, dim=-1, device=device)

with leg_plate, dim_plate:
    init_ideal_point = pyro.sample('theta', dist.Normal(0, 1))
    atten = pyro.sample('tau', dist.Gamma(1, 1))

with poutine.mask(mask=disturb_mask):
    disturb = pyro.sample('v', GaussianRandomWalk(scale=atten.squeeze(), num_steps=max_time.item()))
disturb.shape

with leg_plate, dim_plate:
    with poutine.mask(mask=disturb_mask):
        disturb = pyro.sample('v', GaussianRandomWalk(scale=torch.ones(1), num_steps=max_time.item()))
disturb.shape

with vote_plate, dim_plate:
    polarity = pyro.sample('beta', dist.Normal(0, 5))

with vote_plate:
    popularity = pyro.sample('alpha', dist.Normal(0, 5))

# temp_ideal = init_ideal_point.unsqueeze(-1)
# print(temp_ideal.shape)
# print(atten.shape)
# print(disturb.shape)
# all_temp_ideal = init_ideal_point + (atten * disturb).cumsum(dim=2)
# all_temp_ideal.shape
# print(all_temp_ideal.shape)

ideal_point = pyro.deterministic("ideal_point", torch.cat([init_ideal_point, init_ideal_point + disturb], dim=2))
ideal_point.shape

batch_ideal = ideal_point[legs, :, ind_time_tensor]
polarity_batch = polarity[votes]
popularity_batch = popularity[votes]

batch_ideal.shape
polarity_batch.shape
popularity_batch.shape

torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1).shape
popularity_batch.squeeze().shape

# Combine parameters
logit = torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1) + popularity_batch.squeeze()

obs = responses
obs.shape

if obs is not None:
    # If training outcomes are provided, run the sampling statement for the given data
    with pyro.plate('observe_data', obs.size(0), device=device):
        pyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)
else:
    # If no training outcomes are provided, return the samples from the predicted distributions
    with pyro.plate('observe_data', legs.size(0), device=self.device):
        obs = pyro.sample("obs", dist.Bernoulli(logits=logit))
    return obs


class bayes_dynamic:
    def __init__(self, n_legs, n_votes, max_time, k_dim=1, guide_type="auto", custom_init_values=None, disturb_mask=None, device=None):
        super().__init__()
        self.n_legs = n_legs
        self.n_votes = n_votes
        self.max_time = max_time
        self.k_dim = k_dim
        self.guide_type = guide_type
        self.custom_init_values = custom_init_values
        self.device = device
        self.disturb_mask = disturb_mask

    def model(self, legs, votes, ind_time_tensor, obs=None):
        # Declare context managers, implies conditional independence across the specified dimension
        # 3 Dimensions: legislator/vote/observation, ideology dimension, time dimension
        leg_plate = pyro.plate('legs', n_legs, dim=-3, device=device)
        vote_plate = pyro.plate('votes', n_votes, dim=-3, device=device)
        dim_plate = pyro.plate('dim', k_dim, dim=-2, device=device)
        # time_plate = pyro.plate('time', max_time, dim=-1, device=device)

        with leg_plate, dim_plate:
            init_ideal_point = pyro.sample('theta', dist.Normal(0, 1))
            atten = pyro.sample('tau', dist.Gamma(1, 1))

        # with leg_plate, dim_plate:
        #     if self.disturb_mask is not None:
        #         with poutine.mask(mask=disturb_mask):
        #             disturb = pyro.sample('v', GaussianRandomWalk(scale=torch.ones(1), num_steps=max_time.item()))
        #     else:
        #         disturb = pyro.sample('v', GaussianRandomWalk(scale=torch.ones(1), num_steps=max_time.item()))
        if self.disturb_mask is not None:
            with poutine.mask(mask=disturb_mask):
                disturb = pyro.sample('v', GaussianRandomWalk(scale=atten.squeeze(), num_steps=max_time.item()).to_event())
        else:
            disturb = pyro.sample('v', GaussianRandomWalk(scale=atten.squeeze(), num_steps=max_time.item()).to_event())
        print(f'v shape{disturb.shape}')

        with vote_plate, dim_plate:
            polarity = pyro.sample('beta', dist.Normal(0, 5))

        with vote_plate:
            popularity = pyro.sample('alpha', dist.Normal(0, 5))

        # temp_ideal = init_ideal_point.unsqueeze(-1)
        # print(temp_ideal.shape)
        # print(atten.shape)
        # print(disturb.shape)
        # all_temp_ideal = init_ideal_point + (atten * disturb).cumsum(dim=2)
        # all_temp_ideal.shape
        # print(all_temp_ideal.shape)

        ideal_point = pyro.deterministic("ideal_point", torch.cat([init_ideal_point, init_ideal_point + disturb], dim=2))
        # ideal_point.shape

        batch_ideal = ideal_point[legs, :, ind_time_tensor]
        polarity_batch = polarity[votes]
        popularity_batch = popularity[votes]

        # batch_ideal.shape
        # polarity_batch.shape
        # popularity_batch.shape

        # torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1).shape
        # popularity_batch.squeeze().shape

        # Combine parameters
        logit = torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1) + popularity_batch.squeeze()

        obs = responses
        # obs.shape

        if obs is not None:
            # If training outcomes are provided, run the sampling statement for the given data
            with pyro.plate('observe_data', obs.size(0), device=device):
                pyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)
        else:
            # If no training outcomes are provided, return the samples from the predicted distributions
            with pyro.plate('observe_data', legs.size(0), device=self.device):
                obs = pyro.sample("obs", dist.Bernoulli(logits=logit))
            return obs


    def guide_basic(self, legs, votes, ind_time_tensor, obs=None):
        # register parameters
        loc_theta = pyro.param("loc_init_ideal", lambda: torch.zeros(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1))
        scale_theta = pyro.param("scale_init_ideal", lambda: torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1),
                                 constraint=constraints.positive)
        loc_tau = pyro.param("loc_atten", lambda: torch.ones(self.n_legs, self.k_dim, device=device).unsqueeze(-1),
                             constraint=constraints.positive)
        scale_tau = pyro.param("scale_atten", lambda: torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1),
                               constraint=constraints.positive)
        loc_v = pyro.param("loc_disturb", lambda: torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
        scale_v = pyro.param("scale_disturb", lambda: torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device),
                             constraint=constraints.positive)
        loc_beta = pyro.param("loc_polarity", lambda: torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
        scale_beta = pyro.param("scale_polarity", lambda: 5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1),
                                constraint=constraints.positive)
        loc_alpha = pyro.param("loc_popularity", lambda: torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
        scale_alpha = pyro.param("scale_popularity", lambda: 5.0 * torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1),
                                 constraint=constraints.positive)

        print(f'guide v shape{scale_v.shape}')

        leg_plate = pyro.plate('legs', self.n_legs, dim=-3, device=self.device)
        vote_plate = pyro.plate('votes', self.n_votes, dim=-3, device=self.device)
        dim_plate = pyro.plate('dim', self.k_dim, dim=-2, device=self.device)
        # time_plate = pyro.plate('time', self.max_time, dim=-1, device=self.device)

        with leg_plate, dim_plate:
            pyro.sample('theta', dist.Normal(loc_theta, scale_theta))
            pyro.sample('tau', dist.Gamma(loc_tau, scale_tau))

        if self.disturb_mask is not None:
            with poutine.mask(mask=disturb_mask):
                pyro.sample('v', dist.Normal(loc_v, scale_v))
        else:
            pyro.sample('v', dist.Normal(loc_v, scale_v))

        with vote_plate, dim_plate:
            pyro.sample('beta', dist.Normal(loc_beta, scale_beta))

        with vote_plate:
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
            pyro.get_param_store().get_param("AutoNormal.locs.theta", init_tensor=self.custom_init_values)
            pyro.get_param_store().get_param("AutoNormal.scales.theta", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.tau", init_tensor=0.1 * torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.scales.tau", init_tensor=0.1 * torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.v", init_tensor=torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
            pyro.get_param_store().get_param("AutoNormal.scales.v", init_tensor=torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device), constraint=constraints.interval(1e-7, 1e7))
            pyro.get_param_store().get_param("AutoNormal.locs.beta", init_tensor=torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
            pyro.get_param_store().get_param("AutoNormal.scales.beta", init_tensor=torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            pyro.get_param_store().get_param("AutoNormal.locs.alpha", init_tensor=torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
            pyro.get_param_store().get_param("AutoNormal.scales.alpha", init_tensor=torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1), constraint=constraints.positive)
        elif self.guide_type == "basic":
            pyro.get_param_store().get_param("loc_init_ideal", init_tensor=self.custom_init_values)
            # pyro.get_param_store().get_param("scale_init_ideal", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_atten", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("scale_atten", init_tensor=torch.ones(self.n_legs, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_disturb", init_tensor=torch.zeros(self.n_legs, self.k_dim, self.max_time, device=self.device))
            # pyro.get_param_store().get_param("scale_disturb", init_tensor=torch.ones(self.n_legs, self.k_dim, self.max_time, device=self.device), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_polarity", init_tensor=torch.zeros(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1))
            # pyro.get_param_store().get_param("scale_polarity", init_tensor=5.0 * torch.ones(self.n_votes, self.k_dim, device=self.device).unsqueeze(-1), constraint=constraints.positive)
            # pyro.get_param_store().get_param("loc_popularity", init_tensor=torch.zeros(self.n_votes, 1, device=self.device).unsqueeze(-1))
            # pyro.get_param_store().get_param("scale_popularity", init_tensor=5.0 * torch.ones(self.n_votes, 1, device=self.device).unsqueeze(-1), constraint=constraints.positive)

        logger.info("Run the variational inference")
        # Setup the variational inference
        optim = Adam({'lr': 0.05})
        svi = custom_SVI(self.model, guide, optim, loss=Trace_ELBO(), l2_penalty=0.0)
        self.guide_used = guide

        min_loss = float('inf')  # initialize to infinity
        patience = 0
        for j in tqdm(range(15000)):
            loss = svi.step(legs, votes, ind_time_tensor, responses)
            # loss = svi.penalized_step(legs, votes, ind_time_tensor, responses)

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
test = bayes_dynamic(n_legs, n_votes, max_time, k_dim=k_dim, guide_type="basic", custom_init_values=custom_init_values.unsqueeze(-1), disturb_mask=None, device=device)
test.fit(legs, votes, ind_time_tensor, responses)
pyro.get_param_store().keys()


leg_plate = pyro.plate('legs', n_legs, dim=-3, device=device)
vote_plate = pyro.plate('votes', n_votes, dim=-3, device=device)
dim_plate = pyro.plate('dim', k_dim, dim=-2, device=device)
time_plate = pyro.plate('time', max_time + 1, dim=-1, device=device)

with leg_plate, dim_plate:
    init_ideal_point = pyro.sample('theta', dist.Normal(0, 1))
    atten = pyro.sample('tau', dist.Gamma(0.1, 0.1))

with time_plate:
    disturb = pyro.sample('thingy', dist.Normal(init_ideal_point, atten))

indices = (torch.arange(start=0, end=max_time + 1, device=device).repeat(n_legs, 1))
disturb_mask = indices.le(sessions_served.unsqueeze(-1)).unsqueeze(1)

(disturb * disturb_mask).cumsum(-1)
disturb.cumsum(-1) * disturb_mask

# with poutine.mask(mask=disturb_mask):
#     disturb = pyro.sample('v', GaussianRandomWalk(scale=atten.squeeze(), num_steps=max_time.item()))
# disturb.shape

# with leg_plate, dim_plate:
#     with poutine.mask(mask=disturb_mask):
#         disturb = pyro.sample('v', GaussianRandomWalk(scale=torch.ones(1), num_steps=max_time.item()))
# disturb.shape

with vote_plate, dim_plate:
    polarity = pyro.sample('beta', dist.Normal(0, 5))

with vote_plate:
    popularity = pyro.sample('alpha', dist.Normal(0, 5))

# temp_ideal = init_ideal_point.unsqueeze(-1)
# print(temp_ideal.shape)
# print(atten.shape)
# print(disturb.shape)
# all_temp_ideal = init_ideal_point + (atten * disturb).cumsum(dim=2)
# all_temp_ideal.shape
# print(all_temp_ideal.shape)

# list(map(param_dict.get, legs))

ideal_point = pyro.deterministic("ideal_point", disturb.cumsum(-1))
ideal_point.shape

batch_ideal = ideal_point[legs, :, ind_time_tensor]
polarity_batch = polarity[votes]
popularity_batch = popularity[votes]

batch_ideal.shape
polarity_batch.shape
popularity_batch.shape

torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1).shape
popularity_batch.squeeze().shape

# Combine parameters
logit = torch.sum(batch_ideal * polarity_batch.squeeze(), dim=-1) + popularity_batch.squeeze()

obs = responses
obs.shape

if obs is not None:
    # If training outcomes are provided, run the sampling statement for the given data
    with pyro.plate('observe_data', obs.size(0), device=device):
        pyro.sample("obs", dist.Bernoulli(logits=logit), obs=obs)
else:
    # If no training outcomes are provided, return the samples from the predicted distributions
    with pyro.plate('observe_data', legs.size(0), device=self.device):
        obs = pyro.sample("obs", dist.Bernoulli(logits=logit))
    return obs
