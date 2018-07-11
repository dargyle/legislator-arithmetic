"""
A script to generate random votes with known properties
"""
import os
import numpy as np
import pandas as pd

from scipy import stats

DATA_PATH = os.path.expanduser("~/data/leg_math/")

n_leg = 100
n_bills = 2000
k_dim = 3

include_covariates = True
beta_covar = 0.1

# w = np.repeat(0.5, k_dim)
w = np.array([1.0, 0.5, 0.5])
# Impose sum to one
# w = w / w.sum()
beta = 15.0

cdf_type = "norm"


def generate_random_votes(n_leg=100,
                          n_bills=2000,
                          k_dim=3,
                          include_covariates=True,
                          beta_covar=0.1,
                          w=np.array([1.0, 0.5, 0.5]),
                          beta=15.0,
                          cdf_type="norm"):

    np.random.seed(42)
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
    # Dummy state column
    ideal_points["icpsrState"] = 10
    # Make 1st dimension party
    ideal_points["coord1D"] = ideal_points["coord1D"] * ideal_points["partyCode"].map({100: -1, 200: 1})
    # Randomly assign other dimensions
    if k_dim > 1:
        for i in range(2, k_dim + 1):
            ideal_points[f"coord{i}D"] = ideal_points[f"coord{i}D"] * np.random.choice([-1, 1], size=(n_leg, ))

    # Renorm samples to have max norm of 1
    max_norm = np.sqrt((ideal_points.filter(regex="coord") ** 2).sum(axis=1)).max()
    col_selector = ideal_points.columns.str.contains("coord")
    ideal_points.loc[:, col_selector] = ideal_points.loc[:, col_selector] / max_norm

    # Get random bill points
    bill_ids = "bill_" + pd.Series(np.arange(0, n_bills)).astype(str).str.zfill(len(str(n_bills)))
    yes_points = pd.DataFrame(np.random.uniform(-0.7, 0.7, size=(n_bills, k_dim)),
                              columns=[f"yes_coord{i}D" for i in range(1, k_dim + 1)],
                              index=bill_ids)
    no_points = pd.DataFrame(np.random.uniform(-0.7, 0.7, size=(n_bills, k_dim)),
                             columns=[f"no_coord{i}D" for i in range(1, k_dim + 1)],
                             index=bill_ids)

    # Get all leg/bill pairs
    votes = pd.DataFrame([(i, j) for i in leg_ids for j in bill_ids],
                         columns=["leg_id", "bill_id"])
    # Merge into one dataset
    votes = pd.merge(votes, ideal_points, left_on="leg_id", right_index=True)
    votes = pd.merge(votes, yes_points, left_on="bill_id", right_index=True)
    votes = pd.merge(votes, no_points, left_on="bill_id", right_index=True)

    # Find squared differences in each dimension
    for i in range(1, k_dim + 1):
        votes[f"yes_diff_{i}D"] = (votes[f"coord{i}D"] - votes[f"yes_coord{i}D"]) ** 2
        votes[f"no_diff_{i}D"] = (votes[f"coord{i}D"] - votes[f"no_coord{i}D"]) ** 2

    votes.filter(regex="yes_diff").describe()
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

    votes["party_in_power"] = 100
    votes["congress"] = 114
    bill_numbers = votes["bill_id"].str.split("_").str.get(1).astype(int)
    votes.loc[bill_numbers > int(n_bills / 2), "party_in_power"] = 200
    votes.loc[bill_numbers > int(n_bills / 2), "congress"] = 115
    if include_covariates:
        votes["in_majority"] = 1 * (votes["partyCode"] == votes["party_in_power"])
    else:
        votes["in_majority"] = 0

    if cdf_type == "norm":
        vote_prob = stats.norm.cdf(beta * temp_term + error + beta_covar * votes["in_majority"])
    elif cdf_type == "logit":
        vote_prob = stats.logistic.cdf(beta * temp_term + error + beta_covar * votes["in_majority"])

    votes["vote_prob"] = vote_prob
    votes["vote"] = 1 * (votes["vote_prob"] > 0.5)

    print("Clean data to meet minimum vote conditions")
    min_vote_count = 20
    unanimity_percentage = 0.035

    voter_condition = True
    unanimity_condition = True
    while voter_condition or unanimity_condition:
        print("Testing legislator vote counts")
        leg_vote_counts = votes["leg_id"].value_counts()
        valid_legs = leg_vote_counts[leg_vote_counts > min_vote_count]
        valid_leg_ids = pd.DataFrame(valid_legs.index, columns=["leg_id"])
        n_leg_diff = len(leg_vote_counts) - len(valid_legs)
        if n_leg_diff > 0:
            votes = votes.merge(valid_leg_ids, on="leg_id", how="inner")
        print("Dropped {} legislators with fewer than {} votes".format(n_leg_diff, min_vote_count))
        voter_condition = (n_leg_diff > 0)

        print("Testing unanimity condition")
        vote_percentages = votes.groupby("bill_id")[["vote"]].mean()
        nonunanimous_votes = vote_percentages[(vote_percentages["vote"] < (1 - unanimity_percentage)) &
                                              (vote_percentages["vote"] > unanimity_percentage)]
        nonunanimous_bill_ids = pd.DataFrame(nonunanimous_votes.index)
        n_vote_diff = len(vote_percentages) - len(nonunanimous_bill_ids)
        if n_vote_diff > 0:
            votes = votes.merge(nonunanimous_bill_ids, on="bill_id", how="inner")
        print("Dropped {} votes with fewer than {}% voting in the minority".format(n_vote_diff, unanimity_percentage * 100))
        unanimity_condition = (n_vote_diff > 0)

    # index by leg/bill
    votes = votes.set_index(["leg_id", "bill_id"])

    # Export for use in other esimation
    votes.reset_index().to_feather(DATA_PATH + "/test_votes_df.feather")
    # roll_call.to_csv(DATA_PATH + "/test_votes.csv")
    ideal_points.to_csv(DATA_PATH + "/test_legislators.csv")

    return votes
