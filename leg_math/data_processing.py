import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.expanduser("~/data/leg_math/")


def process_data(data_type="test", congress_cutoff=0, k_dim=1, k_time=0,
                 return_vote_df=False, validation_split=0.2, covariates_list=[]):
    '''Process a dataframe of votes into the format expected by the model

    # Arguments:
        data_type (str): one of ["votes", "test", "cosponsor"] for the kind of
            data to be procesed
        congress_cutoff (int): drop votes from congresses prior to this number
        k_dim (int): the number of dimensions in the model
        k_time (int), EXPERIMENTAL: the number of time dimensions in the model
        return_vote_df (bool): return a data frame of the votes, in addition to
            the model data
        validation_split (float): percentage of the data to keep in the validation set
        covariates_list (list), EXPERIMENTAL: a list of covariate names to
            initialize addition of covariates to the model
    # Returns
        vote_data (dict): a dictionary containing all the data necessary to fit
            the model
        OPTIONAL: vote_df: a pandas dataframe of the votes
    '''
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


if __name__ == '__main__':
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

    print("Get majority party")
    party_majority_rows = party_data.groupby(["congress", "chamber"])["n_members"].idxmax().values
    party_data["majority_party"] = party_data.index.isin(party_majority_rows)
    majority_parties = party_data.loc[party_data["majority_party"], ["congress", "chamber", "party_code"]]
    majority_parties = majority_parties.rename(columns={"party_code": "majority_party_code"})
    leg_data = pd.merge(leg_data, majority_parties, on=["congress", "chamber"])
    leg_data["in_majority"] = 1 * (leg_data["party_code"] == leg_data["majority_party_code"])

    leg_data.columns

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
    vote_df = pd.merge(vote_df, leg_data[["leg_id", "init_value", "in_majority"]].drop_duplicates(), on="leg_id")

    vote_df.to_feather(DATA_PATH + "vote_df_cleaned.feather")
