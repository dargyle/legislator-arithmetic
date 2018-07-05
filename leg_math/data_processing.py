import os
import numpy as np
import pandas as pd

import pickle

DATA_PATH = os.path.expanduser("~/data/leg_math/")

def process_data(data_type="test", congress_cutoff=0, k_dim=1, k_time=0, return_vote_df=False, validation_split=0.2, covariates_list=[]):
    if data_type == "votes":
        vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
    if data_type == "cosponsor":
        # vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data.feather")
        vote_df = pd.read_feather(DATA_PATH + "cosponsor/govtrack_cosponsor_data_smart_oppose.feather")
        vote_df = vote_df.drop("icpsr_id", axis=1)
        sponsor_counts = vote_df.groupby("vote_id")["vote"].sum()
        min_sponsors = 1
        multi_sponsored_bills = sponsor_counts[sponsor_counts >= min_sponsors]
        multi_sponsored_bills.name = "sponsor_counts"
        vote_df = pd.merge(vote_df, multi_sponsored_bills.to_frame(), left_on="vote_id", right_index=True)
    if data_type == "test":
        # roll_call_object = pd.read_csv(DATA_PATH + "/test_votes.csv", index_col=0)
        # vote_df = roll_call_object.replace({1: 1,
        #                                     2: 1,
        #                                     3: 1,
        #                                     4: 0,
        #                                     5: 0,
        #                                     6: 0,
        #                                     7: np.nan,
        #                                     8: np.nan,
        #                                     9: np.nan,
        #                                     0: np.nan})
        # vote_df = vote_df.stack().reset_index()
        # assert not vote_df.isnull().any().any(), "mising codes in votes"
        # vote_df.columns = ["leg_id",  "vote_id", "vote"]
        # vote_df["congress"] = 115
        # vote_df["chamber"] = "s"
        # leg_data = pd.read_csv(DATA_PATH + "/test_legislators.csv", index_col=0)
        # if "partyCode" in leg_data.columns:
        #     leg_data["init_value"] = leg_data["partyCode"].map({100: -1,
        #                                                         200: 1})
        # else:
        #     leg_data["init_value"] = leg_data["party.1"].map({100: -1,
        #                                                       200: 1})
        #
        # # leg_data["init_value"] = 1
        # vote_df = pd.merge(vote_df, leg_data[["init_value"]], left_on="leg_id", right_index=True)
        vote_df = pd.read_feather(DATA_PATH + "/test_votes_df.feather")
        vote_df = vote_df.rename(columns={"bill_id": "vote_id"})
        vote_df["init_value"] = vote_df["partyCode"].map({100: -1, 200: 1})

    if congress_cutoff:
        vote_df = vote_df[vote_df["congress"] >= congress_cutoff]

    first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
    first_session.columns = ["first_session", "last_session"]
    # first_session["first_session"].value_counts()
    vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
    vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]

    leg_ids = vote_df["leg_id"].unique()
    vote_ids = vote_df["vote_id"].unique()

    vote_df_temp = vote_df.copy()
    leg_crosswalk = pd.Series(leg_ids).to_dict()
    leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
    vote_crosswalk = pd.Series(vote_ids).to_dict()
    vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

    vote_df_temp["leg_id"] = vote_df_temp["leg_id"].map(leg_crosswalk_rev)
    vote_df_temp["vote_id"] = vote_df_temp["vote_id"].map(vote_crosswalk_rev)
    # Shuffle the order of the vote data
    # THIS IS IMPORTANT, otherwise will_select just most recent bills
    vote_df_temp = vote_df_temp.sample(frac=1, replace=False, random_state=42)

    init_embedding = vote_df_temp[["leg_id", "init_value"]].drop_duplicates("leg_id").set_index("leg_id").sort_index()

    assert not vote_df_temp.isnull().any().any(), "Missing value in data"

    N = len(vote_df_temp)
    key_index = round(0.2 * N)
    time_passed = [(vote_df_temp["time_passed"] ** i).values for i in range(1, k_time + 1)]

    vote_data = {'J': len(leg_ids),
                 'M': len(vote_ids),
                 'N': N,
                 'j_train': vote_df_temp["leg_id"].values[:(N - key_index)],
                 'j_test': vote_df_temp["leg_id"].values[-key_index:],
                 'm_train': vote_df_temp["vote_id"].values[:(N - key_index)],
                 'm_test': vote_df_temp["vote_id"].values[-key_index:],
                 'y_train': vote_df_temp["vote"].astype(int).values[:(N - key_index)],
                 'y_test': vote_df_temp["vote"].astype(int).values[-key_index:],
                 'time_passed_train': [i[:(N - key_index)] for i in time_passed],
                 'time_passed_test': [i[-key_index:] for i in time_passed],
                 'init_embedding': init_embedding,
                 'vote_crosswalk': vote_crosswalk,
                 'leg_crosswalk': leg_crosswalk,
                 'covariates_train': vote_df_temp[covariates_list].values[:(N - key_index), :],
                 'covariates_test': vote_df_temp[covariates_list].values[-key_index:, :],
                 }

    # Export a pscl rollcall type object of the training data
    if data_type == 'test':
        export_vote_df = vote_df_temp.iloc[:(N - key_index), :]
        export_vote_df = export_vote_df[["leg_id", "vote_id", "vote"]]
        export_vote_df["leg_id"] = export_vote_df["leg_id"].map(leg_crosswalk)
        export_vote_df["vote_id"] = export_vote_df["vote_id"].map(vote_crosswalk)
        roll_call = export_vote_df.set_index(["leg_id", "vote_id"])["vote"].map({1: 1, 0: 6}).unstack()
        roll_call.fillna(9).astype(int).to_csv(DATA_PATH + "/test_votes.csv")

    init_leg_embedding_final = pd.DataFrame(np.random.uniform(-1.0, 1.0, size=(vote_data["J"], k_dim)))
    init_leg_embedding_final.iloc[:, 0] = init_leg_embedding_final.iloc[:, 0].abs() * vote_data["init_embedding"]["init_value"]
    max_norm = np.sqrt((init_leg_embedding_final ** 2).sum(axis=1)).max()
    init_leg_embedding_final = init_leg_embedding_final / (max_norm + 1e-7)

    vote_data['init_embedding'] = init_leg_embedding_final

    if return_vote_df:
        return vote_data, vote_df
    else:
        return vote_data
