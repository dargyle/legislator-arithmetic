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

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

pyro.enable_validation(True)


class wnom_basic(nn.Module):
    """
    A class for implementing wnominate in pytorch
    """
    def __init__(self, n_legs, n_votes, k_dim, pretrained=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom_basic, self).__init__()

        # These are all the wnominate parameters
        if pretrained is not None:
            self.ideal_points = nn.Parameter(pretrained)
        else:
            self.ideal_points = nn.Parameter(torch.rand(n_legs, k_dim))
        self.yes_points = nn.Parameter(torch.rand(n_votes, k_dim))
        self.no_points = nn.Parameter(torch.rand(n_votes, k_dim))

        self.w = nn.Parameter(0.5 * torch.ones((k_dim)))
        self.beta = nn.Parameter(torch.tensor(5.0))

    def max_norm_(self, weight):
        """
        This ensures that the parameters are in a unit hypersphere
        """
        with torch.no_grad():
            norm = weight.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1)
            weight *= desired / norm

    def forward(self, legs, votes):
        """
        Take in the legislator ids and vote ids and generate a prediction
        """
        # Constrain all these things to the a unit hypersphere (ala original NOMINATE)
        self.max_norm_(self.ideal_points)
        self.max_norm_(self.yes_points)
        self.max_norm_(self.no_points)

        # Calculate distances from ideal points
        distances1 = torch.sum(torch.square(self.ideal_points[legs] - self.yes_points[votes]) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(self.ideal_points[legs] - self.no_points[votes]) * torch.square(self.w), axis=1)

        # The key wnominate layer
        result = self.beta * (torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


class wnom_full(nn.Module):
    """
    A class for implementing dwnominate, dynamic ideal points
    """
    def __init__(self, n_legs, n_votes, k_dim, pretrained=None, k_time=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom_full, self).__init__()
        if k_time is not None:
            self.ideal_points = nn.Parameter((torch.rand(n_legs, k_dim, k_time + 1) - 0.5))
        else:
            if pretrained is not None:
                self.ideal_points = nn.Parameter(pretrained)
            else:
                self.ideal_points = nn.Parameter(torch.rand(n_legs, k_dim))
        self.yes_points = nn.Parameter(torch.rand(n_votes, k_dim))
        self.no_points = nn.Parameter(torch.rand(n_votes, k_dim))

        self.w = nn.Parameter(0.5 * torch.ones((k_dim)))
        self.beta = nn.Parameter(torch.tensor(5.0))

    def max_norm_(self, weight):
        """
        This ensures that the parameters are in a unit hypersphere
        """
        with torch.no_grad():
            norm = weight.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1)
            weight *= desired / norm

    def reduced_max_norm_(self, weight, sessions_served=None):
        """
        This ensures that time varying parameters are in the unit hypershpere

        This is trickier with time varying ideal points, the work around here to
        pass information about the number of sessions served and check that the first
        and the last ideal points are in the unit hypersphere. For a linear time trend,
        this is sufficient to ensure all points meet the requirements. For quadratic
        (or higher) time trends, the time trend could leave the hypersphere even if the first
        and last points are valid. Since we're not anticipating using more than linear time trends,
        we won't worry about this for now.
        """
        if sessions_served is not None:
            # This ensures that both the first and last ideal points are in the unit hypersphere
            with torch.no_grad():
                # Check the initial ideal point
                initial_ideal = weight[:, :, 0]
                # Initiate the final ideal point
                final_ideal = torch.clone(initial_ideal)
                # Add in the time trends to obtain the actual final point
                for ii in range(1, weight.shape[2]):
                    final_ideal += weight[:, :, ii] * (sessions_served.unsqueeze(-1) ** ii)
                # Clip the values by the first or last, whichever is bigger
                norm_initial = initial_ideal.norm(2, dim=1, keepdim=True)
                norm_final = final_ideal.norm(2, dim=1, keepdim=True)
                # Get the biger of the two norms
                norm = torch.max(torch.cat([norm_initial, norm_final], axis=1), dim=1, keepdim=True)[0]
                desired = torch.clamp(norm, max=1)
                weight *= (desired / norm).unsqueeze(dim=-1)
        else:
            # If no sessions served information is given, ensure that the point when t=1 is
            # is in the unit hypersphere
            with torch.no_grad():
                reduced_weight = weight.sum(dim=2)
                norm = reduced_weight.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=1)
                weight *= (desired / norm).unsqueeze(dim=-1)

    def forward(self, legs, votes, time_tensor=None, sessions_served=None):
        """
        Take in the legislator and vote ids and generate a prediction
        """

        # Constrain the bill attributes to the a unit hypersphere (ala original NOMINATE)
        self.max_norm_(self.yes_points)
        self.max_norm_(self.no_points)

        # Do the constraint for the time dimension (if present)
        if time_tensor is not None:
            if sessions_served is not None:
                self.reduced_max_norm_(self.ideal_points, sessions_served)
            else:
                self.reduced_max_norm_(self.ideal_points)
            ideal_points_use = torch.sum(self.ideal_points[legs] * time_tensor[votes].unsqueeze(1), axis=2)
        else:
            self.max_norm_(self.ideal_points)
            ideal_points_use = self.ideal_points[legs]

        # Calculate distances from ideal points
        distances1 = torch.sum(torch.square(ideal_points_use - self.yes_points[votes]) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(ideal_points_use - self.no_points[votes]) * torch.square(self.w), axis=1)

        # Final dwnominate layer
        result = self.beta * (torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


if __name__ == '__main__':
    logger.info("Running some basic model tests on synthetic data")
    # Set up environment
    gpu = False
    if gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info("Generate a test dataset that has 2 dimensions")
    random_votes = generate_nominate_votes(n_leg=50, n_votes=100, beta=15.0, beta_covar=0.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
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

    legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
    votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
    responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)

    # Set some constants
    n_legs = len(set(legs.numpy()))
    n_votes = len(set(votes.numpy()))

    logger.info("Set up the pytorch model")
    wnom_model = wnom_full(n_legs, n_votes, k_dim, custom_init_values)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)

    logger.info("Fit the pytorch model")
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

        # if t % 100 == 0:
        #     logger.info(f'epoch {t}, loss: {loss.item()}')

        losses.append(loss.item())
        accuracies.append(accuracy)
        test_losses.append(loss_test.item())
        test_accuracies.append(accuracy_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info("Now try with a time dimension")
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

    logger.info("Setup the pytorch model")
    wnom_model = wnom_full(n_legs, n_votes, k_dim, k_time=k_time)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)

    logger.info("Fit the pytorch model")
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

        # if t % 100 == 0:
        #     logger.info(f'epoch {t}, loss: {loss.item()}')

        losses.append(loss.item())
        accuracies.append(accuracy)
        test_losses.append(loss_test.item())
        test_accuracies.append(accuracy_test)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pd.Series(losses).min()
    pd.Series(losses).idxmin()

    pd.Series(test_losses).min()
    pd.Series(test_losses).idxmin()
