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


def process_data(vote_df, congress_cutoff=0, k_dim=1, k_time=0,
                 return_vote_df=False, validation_split=0.2, covariates_list=[],
                 unanimity_check=True):
    '''Process a dataframe of votes into a dictionary expected by the model

    # Arguments:
        vote_df (DataFrame): A DataFrame of votes to process
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
    '''
    print("Limit the sample")
    if congress_cutoff:
        vote_df = vote_df[vote_df["congress"] >= congress_cutoff].copy()

    if k_time > 0:
        first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
        first_session.columns = ["first_session", "last_session"]
        first_session["sessions_served"] = first_session["last_session"] - first_session["first_session"]
        # first_session["first_session"].value_counts()
        vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
        vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]

    # Shuffle the order of the vote data
    # THIS IS IMPORTANT, otherwise will_select just most recent bills
    vote_df = vote_df.sample(frac=1, replace=False, random_state=42)

    if "vote_weight" not in vote_df.columns:
        vote_df["vote_weight"] = 1.0

    N = len(vote_df)
    key_index = round(validation_split * N)
    print(f"key_index: {key_index}")

    # Keep only votes that are valid in the dataset
    train_data = vote_df.iloc[:(N - key_index), :].copy()
    if unanimity_check:
        train_data = drop_unanimous(train_data, min_vote_count=10, unanimity_percentage=0.001)
    # Ensure test data only contains valid entries
    test_data = vote_df.iloc[(N - key_index):, :]
    test_data = test_data[test_data["leg_id"].isin(train_data["leg_id"])]
    test_data = test_data[test_data["vote_id"].isin(train_data["vote_id"])]

    if k_time > 0:
        time_passed_train = [(train_data["time_passed"] ** i).values for i in range(0, k_time + 1)]
        time_passed_test = [(test_data["time_passed"] ** i).values for i in range(0, k_time + 1)]
    else:
        time_passed_train = []
        time_passed_test = []

    leg_ids = train_data["leg_id"].unique()
    vote_ids = train_data["vote_id"].unique()

    leg_crosswalk = pd.Series(leg_ids).to_dict()
    leg_crosswalk_rev = dict((v, k) for k, v in leg_crosswalk.items())
    vote_crosswalk = pd.Series(vote_ids).to_dict()
    vote_crosswalk_rev = dict((v, k) for k, v in vote_crosswalk.items())

    train_data["leg_id_num"] = train_data["leg_id"].map(leg_crosswalk_rev)
    train_data["vote_id_num"] = train_data["vote_id"].map(vote_crosswalk_rev)
    test_data["leg_id_num"] = test_data["leg_id"].map(leg_crosswalk_rev)
    test_data["vote_id_num"] = test_data["vote_id"].map(vote_crosswalk_rev)

    init_embedding = train_data[["leg_id_num", "init_value"]].drop_duplicates("leg_id_num").set_index("leg_id_num").sort_index()

    assert not vote_df.isnull().any().any(), "Missing value in data"

    vote_data = {'J': len(leg_ids),
                 'M': len(vote_ids),
                 'N': N,
                 'j_train': train_data[["leg_id_num"]].values,
                 'j_test': test_data[["leg_id_num"]].values,
                 'm_train': train_data[["vote_id_num"]].values,
                 'm_test': test_data[["vote_id_num"]].values,
                 'y_train': train_data[["vote"]].astype(int).values,
                 'y_test': test_data[["vote"]].astype(int).values,
                 'time_passed_train': time_passed_train,
                 'time_passed_test': time_passed_test,
                 'init_embedding': init_embedding,
                 'vote_crosswalk': vote_crosswalk,
                 'leg_crosswalk': leg_crosswalk,
                 'covariates_train': train_data[covariates_list].values,
                 'covariates_test': test_data[covariates_list].values,
                 'vote_weight_train': train_data["vote_weight"].values,
                 'vote_weight_test': test_data["vote_weight"].values,
                 'sessions_served': first_session.loc[leg_crosswalk_rev.keys(), "sessions_served"].values,
                 }

    # Export a pscl rollcall type object of the training data
    # if data_type == 'test':
    #     export_vote_df = vote_df.iloc[:(N - key_index), :]
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

    return vote_data


def format_model_data(vote_data, data_params, weight_by_frequency=True):
    '''Format vote_data for use in model, calculate sample weights

    # Args:
        vote_data (dict): A dictionary of processed vote data
        data_params (dict): A parameter dictionary for the model
        weight_by_frequency (bool): Should sample weights be calculated based on frequency
    # Returns:
        x_train (list): A list of input data for the model, training set
        x_test (list): A list of input data for the model, test set
        sample_weights (dict): A dictionary mapping outcomes to sample weights
    '''
    # Weights are probably better, but not in original so comment out here
    if weight_by_frequency:
        sample_weights = (1.0 * vote_data["y_train"].shape[0]) / (len(np.unique(vote_data["y_train"])) * np.bincount(np.squeeze(vote_data["y_train"])))
    else:
        sample_weights = {k: 1 for k in np.unique(vote_data["y_train"])}

    if data_params["covariates_list"]:
        if data_params["k_time"] > 0:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + vote_data["time_passed_train"] + [vote_data["covariates_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + vote_data["time_passed_test"] + [vote_data["covariates_test"]]
        else:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + [vote_data["covariates_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + [vote_data["covariates_test"]]
    else:
        if data_params["k_time"] > 0:
            x_train = [vote_data["j_train"], vote_data["m_train"]] + vote_data["time_passed_train"]
            x_test = [vote_data["j_test"], vote_data["m_test"]] + vote_data["time_passed_test"]
        else:
            x_train = [vote_data["j_train"], vote_data["m_train"]]
            x_test = [vote_data["j_test"], vote_data["m_test"]]
    return x_train, x_test, sample_weights


def prep_r_rollcall(vote_data):
    '''Use the processed data to generate files compatible with the R packages

    Note that the three objects returned are designed to be saved as csv files to work with the
    `test_data_in_r.R` script

    # Args:
        vote_data (DataFrame)
        vote_data (dict): A dictionary of data for the NN-NOMINATE models, prepared by the
            `process_data` function
    # Returns:
        roll_call (DataFrame): Votes processed to match the r package pscl rollcall format
    '''
    train_df_list = [pd.DataFrame(vote_data[k], columns=[k]) for k in ['j_train', 'm_train', 'y_train']]
    train_vote_df = pd.concat(train_df_list, axis=1)
    train_vote_df = train_vote_df.rename(columns={"j_train": "leg_id", "m_train": "vote_id", "y_train": "vote"})
    train_vote_df["leg_id"] = train_vote_df["leg_id"].map(vote_data["leg_crosswalk"])
    train_vote_df["vote_id"] = train_vote_df["vote_id"].map(vote_data["vote_crosswalk"])

    test_df_list = [pd.DataFrame(vote_data[k], columns=[k]) for k in ['j_test', 'm_test', 'y_test']]
    test_vote_df = pd.concat(test_df_list, axis=1)
    test_vote_df = test_vote_df.rename(columns={"j_test": "leg_id", "m_test": "vote_id", "y_test": "vote"})
    test_vote_df["leg_id"] = test_vote_df["leg_id"].map(vote_data["leg_crosswalk"])
    test_vote_df["vote_id"] = test_vote_df["vote_id"].map(vote_data["vote_crosswalk"])

    roll_call = train_vote_df.set_index(["leg_id", "vote_id"])["vote"].map({1: 1, 0: 6}).unstack()
    roll_call = roll_call.fillna(9).astype(int)


    return roll_call
