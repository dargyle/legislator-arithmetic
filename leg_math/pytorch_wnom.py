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


class wnom_basic(nn.Module):
    def __init__(self, num_legs, num_votes, k_dim, pretrained=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom_basic, self).__init__()
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


class wnom_full(nn.Module):
    def __init__(self, num_legs, num_votes, k_dim, pretrained=None, k_time=None):
        """
        Instantiate using the embeddings of legislator and bill info
        """
        super(wnom_full, self).__init__()
        if k_time is not None:
            self.ideal_points = nn.Parameter((torch.rand(num_legs, k_dim, k_time + 1) - 0.5))
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

    def reduced_max_norm_(self, w, sessions_served=None):
        if sessions_served is not None:
            # This ensures that both the first and last ideal points are in the unit hypersphere
            with torch.no_grad():
                # Check the initial ideal point
                # Can vectorize ala the actual model?
                initial_ideal = w[:, :, 0]
                final_ideal = torch.clone(initial_ideal)
                for ii in range(1, w.shape[2]):
                    final_ideal += w[:, :, ii] * (sessions_served.unsqueeze(-1) ** ii)
                norm_initial = initial_ideal.norm(2, dim=1, keepdim=True)
                norm_final = final_ideal.norm(2, dim=1, keepdim=True)
                # Get the biger of the two norms
                norm = torch.max(torch.cat([norm_initial, norm_final], axis=1), dim=1, keepdim=True)[0]
                desired = torch.clamp(norm, max=1)
                w *= (desired / norm).unsqueeze(dim=-1)
        else:
            # This ensures that the second time dim ideal point is in the unit hypersphere
            with torch.no_grad():
                reduced_w = w.sum(dim=2)
                norm = reduced_w.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=1)
                w *= (desired / norm).unsqueeze(dim=-1)

    def forward(self, legs, votes, time_tensor=None, sessions_served=None):
        """
        Take in the legislator and vote ids and generate a prediction
        """

        # Constrain all these things to the a unit hypersphere (ala original NOMINATE)
        self.max_norm_(self.yes_points)
        self.max_norm_(self.no_points)

        if time_tensor is not None:
            if sessions_served is not None:
                self.reduced_max_norm_(self.ideal_points, sessions_served)
            else:
                self.reduced_max_norm_(self.ideal_points)
            ideal_points_use = torch.sum(self.ideal_points[legs] * time_tensor[votes].unsqueeze(1), axis=2)
        else:
            self.max_norm_(self.ideal_points)
            ideal_points_use = self.ideal_points

        distances1 = torch.sum(torch.square(ideal_points_use[legs] - self.yes_points[votes]) * torch.square(self.w), axis=1)
        distances2 = torch.sum(torch.square(ideal_points_use[legs] - self.no_points[votes]) * torch.square(self.w), axis=1)

        result = self.beta * (torch.exp(-0.5 * distances1) - torch.exp(-0.5 * distances2))

        return result


if __name__ == '__main__':


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

    wnom_model = wnom_basic(num_legs, num_votes, k_dim, custom_init_values)

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


    # Add a covariate
    random_votes = generate_nominate_votes(n_leg=100, n_votes=1000, beta=15.0, beta_covar=0.0, k_dim=2, w=np.array([1.0, 1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
    vote_df = random_votes.reset_index()

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

    time_tensor = torch.cat([torch.ones(time_passed.shape[0], device=device).unsqueeze(-1), time_passed], axis=1)
    time_tensor_test = torch.cat([torch.ones(time_passed_test.shape[0], device=device).unsqueeze(-1), time_passed_test], axis=1)

    sessions_served = torch.tensor(vote_data["sessions_served"])

    # Set some constants
    num_legs = len(set(legs.numpy()))
    num_votes = len(set(votes.numpy()))
    # num_covar = covariates.shape[1]


    ideal_points = wnom_model.ideal_points
    ideal_points[legs].shape
    time_tensor[votes].unsqueeze(1).shape
    wnom_model = wnom_full(num_legs, num_votes, k_dim, pretrained=custom_init_values, k_time=1)

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

    wnom_model.ideal_points[torch.arange(0, 100)]
    time_tensor.unsqueeze(-1).shape
    wnom_model.ideal_points[legs] * time_tensor[votes]
    torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2)
    torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2).norm(dim=1)

    wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)

    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)




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
    num_legs = len(set(legs.numpy()))
    num_votes = len(set(votes.numpy()))
    # num_covar = covariates.shape[1]

    wnom_model = wnom_full(num_legs, num_votes, k_dim, pretrained=custom_init_values, k_time=data_params["k_time"])

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

    wnom_model.ideal_points[torch.arange(0, num_legs)]
    time_tensor.unsqueeze(-1).shape
    wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1)
    torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1)
    torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1).norm(dim=1)

    wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)

    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, num_legs)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
    pd.DataFrame(wnom_model.ideal_points[torch.arange(0, num_legs), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
    wnom_model.ideal_points[torch.arange(0, num_legs)].sum(dim=2).norm(dim=1).max()

    wnom_model.beta
    wnom_model.w

    wnom_model.ideal_points.shape
    initial_ideal = wnom_model.ideal_points[torch.arange(0, num_legs), :, 0]
    initial_ideal.shape
    last_session_ideal = initial_ideal + wnom_model.ideal_points[torch.arange(0, num_legs), :, 1] * sessions_served.unsqueeze(-1) + wnom_model.ideal_points[torch.arange(0, num_legs), :, 2] * (sessions_served ** 2).unsqueeze(-1)

    ideal_points = wnom_model.ideal_points[torch.arange(0, num_legs), :, :]
    asdf = torch.clone(initial_ideal)
    for kk in range(1, ideal_points.shape[2]):
        asdf += ideal_points[:, : , kk] * sessions_served.pow(kk).unsqueeze(-1)
    asdf

    last_session_ideal.shape
    pd.Series(last_session_ideal.norm(dim=1).detach().numpy()).hist()
    asdf.norm(dim=1).max()
    pd.DataFrame(asdf.detach().numpy()).plot(kind="scatter", x=0, y=1)

    import torchviz
    from torchviz import make_dot

    make_dot(y_pred)
