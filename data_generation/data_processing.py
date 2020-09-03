import numpy as np
import pandas as pd

from constants import DATA_PATH


def drop_unanimous(vote_df,
                   min_vote_count=20,
                   unanimity_percentage=0.03,
                   ):
    '''Verifies that the votes meet unanimity conditions and legislators meet minimum vote counts

    # Arguments:
        vote_df (DataFrame): A dataframe containing vote data. At a minimum this includes three
            columns (leg_id, vote_id, and vote). Other columns in the DataFrame are retained and
            returned in the result.
        min_vote_count (int): A legislator must have cast at least this many votes to be included in
            the result
        unanimity_percentage (float): A percentage in the interval [0, 1]. If fewer than this
            percentage vote in the minority, the vote is dropped from the result.
    # Returns:
        vote_df (DataFrame): A modified vote DataFrame, with infrequent voters and (near)unanimous
            votes dropped
    '''

    # Set termination conditions prior to loop
    voter_condition = True
    unanimity_condition = True

    # While loop over both the voter condition and the unanimity condition are met
    # Need to ensure that legislators still meet minimum vote count even after unanimous bills
    # are dropped (and vice versa)
    while voter_condition or unanimity_condition:
        print("Testing legislator vote counts")
        # Count the number of votes cast by each legislator id
        leg_vote_counts = vote_df["leg_id"].value_counts()
        # Determine which meet the requirement
        valid_legs = leg_vote_counts[leg_vote_counts > min_vote_count]
        valid_leg_ids = pd.DataFrame(valid_legs.index, columns=["leg_id"])
        n_leg_diff = len(leg_vote_counts) - len(valid_legs)
        # If there is a difference, keep only those that met the condition
        if n_leg_diff > 0:
            vote_df = vote_df.merge(valid_leg_ids, on="leg_id", how="inner")
        print("Dropped {} legislators with fewer than {} votes".format(n_leg_diff, min_vote_count))
        # Update voting condition
        voter_condition = (n_leg_diff > 0)

        print("Testing unanimity condition")
        # Get the percentage voting yea on a given bill
        vote_percentages = vote_df.groupby("vote_id")[["vote"]].mean()
        # Identify votes where the yea percentage is within the unanimity_percentage (for either side)
        nonunanimous_votes = vote_percentages[(vote_percentages["vote"] < (1 - unanimity_percentage)) &
                                              (vote_percentages["vote"] > unanimity_percentage)]
        nonunanimous_vote_ids = pd.DataFrame(nonunanimous_votes.index)
        n_vote_diff = len(vote_percentages) - len(nonunanimous_vote_ids)
        # Keep only nonunanimous votes by merging
        if n_vote_diff > 0:
            vote_df = vote_df.merge(nonunanimous_vote_ids, on="vote_id", how="inner")
        print("Dropped {} votes with fewer than {}% voting in the minority".format(n_vote_diff, unanimity_percentage * 100))
        # Update condition
        unanimity_condition = (n_vote_diff > 0)

    return vote_df


def process_data(data_type="test", congress_cutoff=0, k_dim=1, k_time=0,
                 return_vote_df=False, validation_split=0.2, covariates_list=[],
                 unanimity_check=True):
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
    # Returns:
        vote_data (dict): a dictionary containing all the data necessary to fit
            the model
        OPTIONAL: vote_df: a pandas dataframe of the votes
    '''
    print("Load raw data")
    if isinstance(data_type, str):
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
    else:
        # Assume that the data_type argument is a vote dataframe
        vote_df = data_type

    print("Limit the sample")
    if congress_cutoff:
        vote_df = vote_df[vote_df["congress"] >= congress_cutoff]

    if k_time > 0:
        first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
        first_session.columns = ["first_session", "last_session"]
        # first_session["first_session"].value_counts()
        vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
        vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]

    leg_ids = vote_df["leg_id"].unique()
    vote_ids = vote_df["vote_id"].unique()

    if return_vote_df:
        vote_df_temp = vote_df.copy()
    else:
        # Avoid unecessary data copy if not returning raw data
        vote_df_temp = vote_df
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

    if "vote_weight" not in vote_df_temp.columns:
        vote_df_temp["vote_weight"] = 1.0

    assert not vote_df_temp.isnull().any().any(), "Missing value in data"

    N = len(vote_df_temp)
    key_index = round(validation_split * N)
    print(f"key_index: {key_index}")

    # Keep only votes that are valid in the dataset
    train_data = vote_df_temp.iloc[:(N - key_index), :]
    if unanimity_check:
        train_data = drop_unanimous(train_data, min_vote_count=10, unanimity_percentage=0.001)
    # Ensure test data only contains valid entries
    test_data = vote_df_temp.iloc[(N - key_index):, :]
    test_data = test_data[test_data["leg_id"].isin(train_data["leg_id"])]
    test_data = test_data[test_data["vote_id"].isin(train_data["vote_id"])]

    time_passed_train = [(train_data["time_passed"] ** i).values for i in range(1, k_time + 1)]
    time_passed_test = [(test_data["time_passed"] ** i).values for i in range(1, k_time + 1)]

    vote_data = {'J': len(leg_ids),
                 'M': len(vote_ids),
                 'N': N,
                 'j_train': train_data["leg_id"].values,
                 'j_test': test_data["leg_id"].values,
                 'm_train': train_data["vote_id"].values,
                 'm_test': test_data["vote_id"].values,
                 'y_train': train_data["vote"].astype(int).values,
                 'y_test': test_data["vote"].astype(int).values,
                 'time_passed_train': time_passed_train,
                 'time_passed_test': time_passed_test,
                 'init_embedding': init_embedding,
                 'vote_crosswalk': vote_crosswalk,
                 'leg_crosswalk': leg_crosswalk,
                 'covariates_train': train_data[covariates_list].values,
                 'covariates_test': test_data[covariates_list].values,
                 'vote_weight_train': train_data["vote_weight"].values,
                 'vote_weight_test': test_data["vote_weight"].values,
                 }

    # Export a pscl rollcall type object of the training data
    # if data_type == 'test':
    #     export_vote_df = vote_df_temp.iloc[:(N - key_index), :]
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

    # TODO: Refactor to ditch the dual return
    if return_vote_df:
        return vote_data, vote_df
    else:
        return vote_data