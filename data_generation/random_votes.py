"""
A script to generate random votes with known properties
"""
import os
import numpy as np
import pandas as pd

from scipy import stats

from data_generation.data_processing import drop_unanimous, process_data, prep_r_rollcall

from constants import DATA_PATH

SYNTHETIC_PATH = DATA_PATH + 'synthetic/'
if not os.path.exists(SYNTHETIC_PATH):
    os.makedirs(SYNTHETIC_PATH)


def generate_nominate_votes(n_leg=100,
                            n_votes=2000,
                            k_dim=3,
                            w=np.array([1.0, 0.5, 0.5]),
                            beta=15.0,
                            beta_covar=0.0,
                            cdf_type="norm",
                            drop_unanimous_votes=True,
                            replication_seed=None):
    '''Generate synthetic votes assuming a NOMINATE framework with the specified parameters.



    # Arguments:
        n_leg (int): The number of voters in the synthetic data
        n_votes (int): The number of votes in the synthetic data
        k_dim (int): The number of dimensions in the data generating process
        beta_covar (float): Parameter for the covariates. The default value of 0 implies no
            covariates will be included. If non-zero, generates a single covariate called
            "in_majority", which is designed to switch between two sessions.
        w (np.array): A vector containing the NOMINATE parameter weights for each dimension
        beta (float): The parameter weight for the combined NOMINATE term
        cdf_type (str): The cdf function to use, supported values are "norm" and "logit"
        drop_unanimous_votes (bool): If True (default), drop any randomly generated unanimous votes
        replication_run (int): If None (default), votes are generated randomly. If a value is given,
            this is used as to set the random seed for replication.
    # Returns:
        vote_df (DataFrame): A DataFrame of synthetic votes, indexed by leg_id and vote_id. This
            DataFrame also contains the underlying values necessary to arrive at the prediciton.

    '''
    assert len(w) == k_dim, "the length of w and k_dim must match!"

    # Set seed for replicability
    if replication_seed:
        np.random.seed(replication_seed)
    # Impose sum to one
    # w = w / w.sum()
    # Get random ideal points
    leg_ids = "leg_" + pd.Series(np.arange(0, n_leg)).astype(str).str.zfill(len(str(n_leg)))
    leg_columns = [f"coord{i}D" for i in range(1, k_dim + 1)]
    ideal_points = pd.DataFrame(np.random.uniform(size=(n_leg, k_dim)),
                                columns=leg_columns,
                                index=leg_ids)
    ideal_points.index.name = "leg_id"
    # Randomly assign party
    ideal_points["partyCode"] = np.random.choice([100, 200], size=(n_leg,))
    # Definte initial values  based on party
    ideal_points["init_value"] = ideal_points["partyCode"].map({100: -1, 200: 1})
    # Dummy state column
    ideal_points["icpsrState"] = 10
    # Make 1st dimension party
    ideal_points["coord1D"] = ideal_points["coord1D"] * ideal_points["init_value"]
    # Randomly assign other dimensions
    if k_dim > 1:
        for i in range(2, k_dim + 1):
            ideal_points[f"coord{i}D"] = ideal_points[f"coord{i}D"] * np.random.choice([-1, 1], size=(n_leg, ))

    # Renorm samples to have max norm of 1
    max_norm = np.sqrt((ideal_points.filter(regex="coord") ** 2).sum(axis=1)).max()
    col_selector = ideal_points.columns.str.contains("coord")
    ideal_points.loc[:, col_selector] = ideal_points.loc[:, col_selector] / max_norm

    # Get random bill points
    vote_ids = "bill_" + pd.Series(np.arange(0, n_votes)).astype(str).str.zfill(len(str(n_votes)))
    yes_points = pd.DataFrame(np.random.uniform(-0.7, 0.7, size=(n_votes, k_dim)),
                              columns=[f"yes_coord{i}D" for i in range(1, k_dim + 1)],
                              index=vote_ids)
    no_points = pd.DataFrame(np.random.uniform(-0.7, 0.7, size=(n_votes, k_dim)),
                             columns=[f"no_coord{i}D" for i in range(1, k_dim + 1)],
                             index=vote_ids)

    # Get all leg/bill pairs
    votes = pd.DataFrame([(i, j) for i in leg_ids for j in vote_ids],
                         columns=["leg_id", "vote_id"])
    # Merge into one dataset
    votes = pd.merge(votes, ideal_points, left_on="leg_id", right_index=True)
    votes = pd.merge(votes, yes_points, left_on="vote_id", right_index=True)
    votes = pd.merge(votes, no_points, left_on="vote_id", right_index=True)

    # Find squared differences in each dimension
    for i in range(1, k_dim + 1):
        votes[f"yes_diff_{i}D"] = (votes[f"coord{i}D"] - votes[f"yes_coord{i}D"]) ** 2
        votes[f"no_diff_{i}D"] = (votes[f"coord{i}D"] - votes[f"no_coord{i}D"]) ** 2

    yes_term = (votes.filter(regex="yes_diff") * w**2).sum(axis=1)
    no_term = (votes.filter(regex="no_diff") * w**2).sum(axis=1)

    temp_term = np.exp(-0.5 * yes_term) - np.exp(-0.5 * no_term)

    # Randomly assign an error term
    # Per DW-NOMINATE beta = 1 / sigma^2
    # NOTE: Unclear if I should have a single error term or one for each or yes/no
    # Doesn't mattter much either way (sum or normals is normal, small change to sd)
    # yes_error = np.random.normal(0, np.sqrt(1 / beta), size=len(votes))
    # no_error = np.random.normal(0, np.sqrt(1 / beta), size=len(votes))
    # error = yes_error + no_error
    # NOTE: Turns out that the assumption is on the difference of the error terms
    error = np.random.normal(0, np.sqrt(1 / beta), size=len(votes))

    # Generate a covariate for being a member of the party in power
    votes["party_in_power"] = 100
    votes["congress"] = 114
    bill_numbers = votes["vote_id"].str.split("_").str.get(1).astype(int)
    votes.loc[bill_numbers > int(n_votes / 2), "party_in_power"] = 200
    votes.loc[bill_numbers > int(n_votes / 2), "congress"] = 115
    if beta_covar != 0.0:
        votes["in_majority"] = 1 * (votes["partyCode"] == votes["party_in_power"])
    else:
        votes["in_majority"] = 0

    if cdf_type == "norm":
        vote_prob = stats.norm.cdf(beta * temp_term + error + beta_covar * votes["in_majority"])
    elif cdf_type == "logit":
        vote_prob = stats.logistic.cdf(beta * temp_term + error + beta_covar * votes["in_majority"])

    votes["vote_prob"] = vote_prob
    votes["vote"] = 1 * (votes["vote_prob"] > 0.5)

    if drop_unanimous_votes:
        print("Clean data to meet minimum vote conditions")
        votes = drop_unanimous(votes, min_vote_count=20, unanimity_percentage=0.03)

    # index by leg/bill
    votes = votes.set_index(["leg_id", "vote_id"])

    return votes


if __name__ == '__main__':
    votes = generate_nominate_votes(beta=5.0, cdf_type="norm", replication_seed=42)
    votes.reset_index().to_feather(SYNTHETIC_PATH + "/test_votes_df_norm.feather")

    votes = generate_nominate_votes(beta=5.0, cdf_type="logit", replication_seed=42)
    votes.reset_index().to_feather(SYNTHETIC_PATH + "/test_votes_df_logit.feather")

    votes = generate_nominate_votes(beta=5.0, beta_covar=5.0, cdf_type="logit", replication_seed=42)
    votes.reset_index().to_feather(SYNTHETIC_PATH + "/test_votes_df_covar.feather")

    vote_df = pd.read_feather(SYNTHETIC_PATH + "/test_votes_df_norm.feather")

    vote_data = process_data(vote_df)
    roll_call = prep_r_rollcall(vote_data)

    leg_info_cols = ["leg_id", "partyCode", "icpsrState"] + [f"coord{i}D" for i in range(1, 4)]
    leg_data = vote_df[leg_info_cols].drop_duplicates()

    vote_info_cols = ["vote_id", "congress"] + [f"yes_coord{i}D" for i in range(1, 4)] + [f"no_coord{i}D" for i in range(1, 4)]
    vote_metadata = vote_df[vote_info_cols].drop_duplicates()

    roll_call.to_csv(SYNTHETIC_PATH + "/test_votes.csv")
    leg_data.to_csv(SYNTHETIC_PATH + "/test_legislators.csv", index=False)
    vote_metadata.to_csv(SYNTHETIC_PATH + "/test_vote_metadata.csv", index=False)
