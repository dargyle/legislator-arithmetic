import os
import numpy as np
import pandas as pd

from scipy import stats

DATA_PATH = os.path.expanduser("~/data/leg_math/")

n_leg = 100
n_bills = 1000
k_dim = 3

# w = np.repeat(0.5, k_dim)
w = np.array([1.5, 0.5, 0.5])
# Impose sum to one
w = w / w.sum()
beta = 15.0

cdf_type = "norm"

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
max_norm = metric = np.sqrt((ideal_points.filter(regex="coord") ** 2).sum(axis=1)).max()
ideal_points = ideal_points / max_norm

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
# index by leg/bill
votes = votes.set_index(["leg_id", "bill_id"])

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

if cdf_type == "norm":
    vote_prob = stats.norm.cdf(beta * temp_term + error)
elif cdf_type == "logit":
    vote_prob = stats.logistic.cdf(beta * temp_term + error)

votes["vote_prob"] = vote_prob
votes["vote"] = 1 * (votes["vote_prob"] > 0.5)

# Map to Poole/Rosenthal recordings
roll_call = votes["vote"].map({1: 1, 0: 6}).unstack()

# Export for use in other esimation
roll_call.to_csv(DATA_PATH + "/test_votes.csv")
ideal_points.to_csv(DATA_PATH + "/test_legislators.csv")
