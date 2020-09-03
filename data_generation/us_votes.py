"""
A script to download and save US voting data
"""

import numpy as np
import pandas as pd

from data_generation.data_processing import drop_unanimous

from constants import DATA_PATH

print("Load raw files")
renew_files = True
if renew_files:
    # If renew_files is True, download the data again from the original source
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
    # Else, rely on local cached files
    vote_metadata = pd.read_feather(DATA_PATH + "vote_metadata.feather")
    roll_call_votes = pd.read_feather(DATA_PATH + "roll_call_votes.feather")
    leg_data = pd.read_feather(DATA_PATH + "leg_data.feather")
    party_data = pd.read_feather(DATA_PATH + "party_data.feather")

assert not roll_call_votes.duplicated(["leg_id", "vote_id"]).any(), "Duplicated leg/vote pairs!"

leg_party = leg_data[["leg_id", "party_code"]].drop_duplicates()
# leg_party[leg_party["leg_id"].duplicated(keep=False)].sort_values("leg_id")

print("Get majority party")
# Identify the majority party by finding the most common party in a given session
# Note that this doesn't deal with ties and may not be correct for other reasons (e.g. retirements)
# However, it's a reasonable first pass
party_majority_rows = party_data.groupby(["congress", "chamber"])["n_members"].idxmax().values
party_data["majority_party"] = party_data.index.isin(party_majority_rows)
majority_parties = party_data.loc[party_data["majority_party"], ["congress", "chamber", "party_code"]]
majority_parties = majority_parties.rename(columns={"party_code": "majority_party_code"})
leg_data = pd.merge(leg_data, majority_parties, on=["congress", "chamber"])
leg_data["in_majority"] = 1 * (leg_data["party_code"] == leg_data["majority_party_code"])


print("Map the vote categories")
# These are standard voteview codes (see earlier link to documentation)
yea_codes = [1, 2, 3]
nay_codes = [4, 5, 6]
present_codes = [7, 8]
not_voting_codes = [0, 9]
roll_call_votes["vote"] = np.nan
roll_call_votes.loc[roll_call_votes["cast_code"].isin(yea_codes), "vote"] = 1
roll_call_votes.loc[roll_call_votes["cast_code"].isin(nay_codes), "vote"] = 0

print("Subset to valid votes")
# Any votes that are still np.nan are dropped
vote_df = roll_call_votes.dropna(subset=["vote"]).copy().reset_index(drop=True)
assert not vote_df.duplicated(["leg_id", "vote_id"]).any(), "Duplicated leg/vote pairs!"
# Save the raw votes file
vote_df.to_feather(DATA_PATH + "vote_df_raw.feather")

print("Clean data to meet minimum vote conditions")
vote_df = drop_unanimous(vote_df)

assert not vote_df.duplicated(["leg_id", "vote_id"]).any(), "Duplicated leg/vote pairs!"

# Use party set initial values
# Ensures that primary axis primarily aligns with R on right and D on left
leg_data["init_value"] = 0
leg_data.loc[leg_data["party_code"] == 100, "init_value"] = -1
leg_data.loc[leg_data["party_code"] == 200, "init_value"] = 1
merge_leg_data = leg_data[["congress", "leg_id", "init_value", "in_majority"]].drop_duplicates(["congress", "leg_id"])
vote_df = pd.merge(vote_df, merge_leg_data, on=["congress", "leg_id"])
assert not vote_df.duplicated(["leg_id", "vote_id"]).any(), "Duplicated leg/vote pairs!"

# Save the cleaned votes
vote_df.to_feather(DATA_PATH + "vote_df_cleaned.feather")

# len(vote_df["vote_id"].unique())
