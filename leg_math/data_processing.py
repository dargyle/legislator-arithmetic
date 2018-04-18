import os
import numpy as np
import pandas as pd

import pickle

renew_files = True
DATA_PATH = os.path.expanduser("~/data/leg_math/")

print("Load raw files")
if renew_files:
    # Documentation at: https://voteview.com/static/docs/csv_docs.html
    vote_metadata = pd.read_csv("https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
                                usecols=["congress", "chamber", "rollnumber", "date", "session", "bill_number"])
    roll_call_votes = pd.read_csv("https://voteview.com/static/data/out/votes/HSall_votes.csv",
                                  usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"])
    leg_data = pd.read_csv("https://voteview.com/static/data/out/members/HSall_members.csv")
    party_data = pd.read_csv("https://voteview.com/static/data/out/parties/HSall_parties.csv")

    roll_call_votes = roll_call_votes.rename(columns={"icpsr": "leg_id"})
    leg_data = leg_data.rename(columns={"icpsr": "leg_id"})

    vote_metadata["vote_id"] = (vote_metadata["congress"].astype(str) +
                                vote_metadata["chamber"].str.slice(0, 1).str.lower() + "_" +
                                vote_metadata["rollnumber"].astype(str).str.zfill(4))
    roll_call_votes["vote_id"] = (roll_call_votes["congress"].astype(str) +
                                  roll_call_votes["chamber"].str.slice(0, 1).str.lower() + "_" +
                                  roll_call_votes["rollnumber"].astype(str).str.zfill(4))

    vote_metadata.to_feather(DATA_PATH + "vote_metadata.feather")
    roll_call_votes.to_feather(DATA_PATH + "roll_call_votes.feather")
    leg_data.to_feather(DATA_PATH + "leg_data.feather")
    party_data.to_feather(DATA_PATH + "party_data.feather")
else:
    vote_metadata = pd.read_feather(DATA_PATH + "vote_metadata.feather")
    roll_call_votes = pd.read_feather(DATA_PATH + "roll_call_votes.feather")
    leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")
    party_data = pd.read_feather(DATA_PATH + "party_data.feather")

leg_party = leg_data[["leg_id", "party_code"]].drop_duplicates()
leg_party[leg_party["leg_id"].duplicated(keep=False)].sort_values("leg_id")

print("Map the vote categories")

yea_codes = [1, 2, 3]
nay_codes = [4, 5, 6]
present_codes = [7, 8]
not_voting_codes = [0, 9]
roll_call_votes["vote"] = np.nan
roll_call_votes.loc[roll_call_votes["cast_code"].isin(yea_codes), "vote"] = 1
roll_call_votes.loc[roll_call_votes["cast_code"].isin(nay_codes), "vote"] = 0

print("Subset to valid votes")
vote_df = roll_call_votes.dropna(subset=["vote"]).copy().reset_index(drop=True)
vote_df.to_feather(DATA_PATH + "vote_df_raw.feather")
# initial_congress = 110
# vote_df = vote_df[vote_df["congress"] >= initial_congress].copy()

print("Clean data to meet minimum vote conditions")
min_vote_count = 20
unanimity_percentage = 0.03

voter_condition = True
unanimity_condition = True
while voter_condition or unanimity_condition:
    print("Testing legislator vote counts")
    leg_vote_counts = vote_df["leg_id"].value_counts()
    valid_legs = leg_vote_counts[leg_vote_counts > min_vote_count]
    valid_leg_ids = pd.DataFrame(valid_legs.index, columns=["leg_id"])
    n_leg_diff = len(leg_vote_counts) - len(valid_legs)
    if n_leg_diff > 0:
        vote_df = vote_df.merge(valid_leg_ids, on="leg_id", how="inner")
    print("Dropped {} legislators with fewer than {} votes".format(n_leg_diff, min_vote_count))
    voter_condition = (n_leg_diff > 0)

    print("Testing unanimity condition")
    vote_percentages = vote_df.groupby("vote_id")[["vote"]].mean()
    nonunanimous_votes = vote_percentages[(vote_percentages["vote"] < (1 - unanimity_percentage)) &
                                          (vote_percentages["vote"] > unanimity_percentage)]
    nonunanimous_vote_ids = pd.DataFrame(nonunanimous_votes.index)
    n_vote_diff = len(vote_percentages) - len(nonunanimous_vote_ids)
    if n_vote_diff > 0:
        vote_df = vote_df.merge(nonunanimous_vote_ids, on="vote_id", how="inner")
    print("Dropped {} votes with fewer than {}% voting in the minority".format(n_vote_diff, unanimity_percentage * 100))
    unanimity_condition = (n_vote_diff > 0)

leg_data["init_value"] = 0
leg_data.loc[leg_data["party_code"] == 100, "init_value"] = -1
leg_data.loc[leg_data["party_code"] == 200, "init_value"] = 1
vote_df = pd.merge(vote_df, leg_data[["leg_id", "init_value"]].drop_duplicates(), on="leg_id")

vote_df.to_feather(DATA_PATH + "vote_df_cleaned.feather")

# vote_df_temp = vote_df.copy()
# leg_ids = vote_df_temp["leg_id"].unique()
# vote_ids = vote_df_temp["vote_id"].unique()
#
# leg_crosswalk = pd.Series(leg_ids).to_dict()
# leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
# vote_crosswalk = pd.Series(vote_ids).to_dict()
# vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())
#
# vote_df_temp["leg_id"] = vote_df_temp["leg_id"].map(leg_crosswalk_rev)
# vote_df_temp["vote_id"] = vote_df_temp["vote_id"].map(vote_crosswalk_rev)
# # Shuffle the order of the vote data
# vote_df_temp = vote_df_temp.sample(frac=1, replace=False)
#
# vote_df_temp.to_feather(DATA_PATH + "vote_df_final.feather")

# init_embedding = vote_df_temp[["leg_id", "init_value"]].drop_duplicates().set_index("leg_id").sort_index()

# vote_data = {'J': len(leg_ids),
#              'M': len(vote_ids),
#              'N': len(vote_df_temp),
#              'j': vote_df_temp["leg_id"].values,
#              'm': vote_df_temp["vote_id"].values,
#              'y': vote_df_temp["vote"].astype(int).values,
#              'init_embedding': init_embedding,
#              'vote_crosswalk': vote_crosswalk,
#              'leg_crosswalk': leg_crosswalk}

# print("N Legislators: {}".format(len(leg_ids)))
# print("N Votes: {}".format(len(vote_ids)))
# print("N: {}".format(len(vote_df_temp)))

# with open(DATA_PATH + 'data_bundle.pkl', 'wb') as f:
#     pickle.dump(vote_data, f)
