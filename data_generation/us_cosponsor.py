import os
import numpy as np
import pandas as pd

from requests import get
from io import BytesIO
from zipfile import ZipFile

import igraph

from tqdm import tqdm

from constants import DATA_PATH

COSPONSOR_PATH = DATA_PATH + "cosponsor/"
if not os.path.exists(COSPONSOR_PATH):
    os.makedirs(COSPONSOR_PATH)


def get_leg_info():
    ''' This function loads data about all current and former legislators into a DataFrame

    # Args:
        None
    # Returns:
        leg_data (DataFrame): A DataFrame of legislator data
    '''
    current_leg = pd.read_csv("https://theunitedstates.io/congress-legislators/legislators-current.csv")
    historical_leg = pd.read_csv("https://theunitedstates.io/congress-legislators/legislators-historical.csv")
    leg_data = pd.concat([historical_leg, current_leg], ignore_index=True)
    return leg_data


# congress_num = 105
# bill_id = 'sconres110-105'
def generate_nays(bill_id, n_nays, vote_df, h_list, s_list):
    '''Generate dissenting votes from the set of non-sponsors

    # Args:
        bill_id (str): The bill id to generate the nay list for
        n_nays (int): The number of opposition votes to generate
        vote_df (DataFrame): The dataset consisting of all sponsorship choices
        _list (list): A list of all possible house members
        s_list (list): A list of all possible senators
    # Returns:
        result_df (DataFrame): A set of no votes for a given bill id
    '''
    # print(bill_id)
    # Subset the data to only the bill of interest
    bill_sponsor_df = vote_df[vote_df["bill_number"] == bill_id]
    if bill_id[0] == "h":
        nays_options = np.setdiff1d(h_list, bill_sponsor_df["thomas_id"])
    if bill_id[0] == "s":
        nays_options = np.setdiff1d(s_list, bill_sponsor_df["thomas_id"])
    if nays_options.size > 0:
        nays_ids = np.random.choice(nays_options, size=min(n_nays, len(nays_options)), replace=False)
        result_df = pd.DataFrame(nays_ids, columns=["thomas_id"])
        result_df["bill_number"] = bill_id
    else:
        result_df = pd.DataFrame()
    return result_df


def get_personalized_pageranks(sponsor_data,
                               chamber_letter='s',
                               n_nays=10,
                               upweight_primary_sponsor=True,
                               min_bills_sponsored=25,
                               use_inverse=True):
    '''Generate "No" votes in a more sophisticated

    Create the cosponsorship network of the entire legislative chamber. Calculate the personalized pagerank
    of the entire sponsor list, randomly choose as nays using the pagerank distribution.

    # Args:
        sponsor_data (DataFrame): The complete DataFrame of sponsor information
        chamber_letter (str): The chamber under consideration, 'h' for house and 's' for senate
        n_nays (int): Number of no votes to populate
        upweight_primary_sponsor (bool): Give the primary sponsor extra weight in the network
        min_bills_sponsored (int): Minimum number of sponsored bills to be included in the analysis
        use_inverse (bool): Pagerank gives highest values to the "closest" nodes, meaning the highest
            probability of selection goes to legislators who were most likely to cosponsor but did
            not. This argument inverts the pagerank weights, such that those most distant are most
            likely to be randomly selected.
    # Returns:
        vote_df_temp (DataFrame): A DataFrame of predicted nays
    '''
    # Subset the data to only active sponsors and only the chamber under consideration
    sponsor_cols = ["thomas_id", "bill_number", "sponsor"]
    bill_sponsors = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), sponsor_cols].copy()
    bill_sponsors = bill_sponsors[bill_sponsors["bill_number"].str.slice(0, 1) == chamber_letter]

    # Ensure that legislators meet the minimum sponsorship requirement
    n_bill_sponsors = bill_sponsors["thomas_id"].value_counts()
    valid_leg_ids = n_bill_sponsors[n_bill_sponsors > min_bills_sponsored].index.tolist()
    bill_sponsors = bill_sponsors[bill_sponsors["thomas_id"].isin(valid_leg_ids)]

    if upweight_primary_sponsor:
        # This divides "credit" between sponsors and the set of copsonsors by sponsor type
        # For example, a bill with one primary sponsor and 3 cosponsors the primary sponsor will
        # get half weight and the cosponsors will each get 1/6 (one-third of one-half)
        link_weights = 1.0 / bill_sponsors.groupby(["bill_number", "sponsor"])[["thomas_id"]].count()
        link_weights.columns = ["link_weights"]
        bill_sponsors = pd.merge(bill_sponsors, link_weights, left_on=["bill_number", "sponsor"], right_index=True)
        bill_sponsors.groupby("bill_number").sum()
        # Ensure that that weights sum to 1 by accounting for number of sponsor types
        # This also fixes the problem of solo sponsored bills getting half weight, but they don't
        # matter for the network anyway
        sponsor_type_counts = link_weights.index.get_level_values("bill_number").value_counts().to_frame(name="sponsor_type_counts")
        bill_sponsors = pd.merge(bill_sponsors, sponsor_type_counts, left_on="bill_number", right_index=True)
        bill_sponsors["link_weights"] = bill_sponsors["link_weights"] / bill_sponsors["sponsor_type_counts"]
    else:
        # Evenly assign "credit" to all people on a bill
        link_weights = 1.0 / bill_sponsors.groupby("bill_number")[["leg_id"]].count()
        link_weights.columns = ["link_weights"]
        bill_sponsors = pd.merge(bill_sponsors, link_weights, left_on="bill_number", right_index=True)

    assert np.allclose(bill_sponsors.groupby("bill_number")["link_weights"].sum().values, 1.0), "bill weights don't sum to 1"

    # Generate a network based on the sponsor data
    g_bill_sponsor = igraph.Graph.TupleList(bill_sponsors[["bill_number", "thomas_id", "link_weights"]].values.tolist(), weights=True)
    # Calculated overall centrality (not needed)
    # overall_centrality = pd.DataFrame(g_bill_sponsor.personalized_pagerank(weights="weight", damping=0.5),
    #                                   index=g_bill_sponsor.vs["name"], columns=["centrality"])
    # leg_centrality = overall_centrality[~overall_centrality.index.isin(bill_sponsors["bill_number"])]

    # Get a list of all bill ids
    bill_ids = bill_sponsors["bill_number"].unique()

    inferred_vote_list = []
    # Generate inferred nays for all bill ids in the data
    for bill_id in tqdm(bill_ids):
        # print(bill_id)
        # Get the set of sponsors for a bill and set them as the reset values in the personalized
        # pagerank calculation. Reset values are expected as probability vector for each vertex
        # in the network. We want sponsors to get the weights calculated above, all other nodes
        # (legislator or bill) should get 0.
        specific_sponsors = bill_sponsors.loc[bill_sponsors["bill_number"] == bill_id, ["thomas_id", "link_weights"]].set_index("thomas_id")
        reset_values = specific_sponsors.reindex(g_bill_sponsor.vs["name"], fill_value=0)["link_weights"]

        # Verify the reset values are a probability distribution
        assert np.allclose(reset_values.sum(), 1.0), "invalid reset distribution"

        # Calculate pageranks and put them into a DataFrame, dropping all bill nodes
        pageranks = g_bill_sponsor.personalized_pagerank(weights="weight", damping=0.75, reset=reset_values.values.tolist())
        bill_centrality = pd.DataFrame(pageranks, index=g_bill_sponsor.vs["name"], columns=["centrality"])
        bill_centrality = bill_centrality[~bill_centrality.index.isin(bill_sponsors["bill_number"])]

        # Normalize the resulting bill centrality to sum to 1
        bill_centrality = bill_centrality / bill_centrality.sum()

        if bill_centrality.size > 0:
            n_possible_nays = len(bill_centrality)

            if use_inverse:
                inverse_centrality = 1 - bill_centrality
                inverse_centrality = inverse_centrality / inverse_centrality.sum()
                nays_ids = inverse_centrality.sample(n=min(n_nays, n_possible_nays), weights="centrality", replace=False).index.tolist()
            else:
                nays_ids = bill_centrality.sample(n=min(n_nays, n_possible_nays), weights="centrality", replace=False).index.tolist()
            result_df = pd.DataFrame(nays_ids, columns=["thomas_id"])
            result_df["bill_number"] = bill_id
        else:
            result_df = pd.DataFrame()
        inferred_vote_list += [result_df]

    # Combine the resulting data
    nay_df = pd.concat(inferred_vote_list)
    vote_df = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), ["thomas_id", "bill_number"]].copy()
    clear_nays = sponsor_data.loc[sponsor_data["date_withdrawn"].notnull(), ["thomas_id", "bill_number"]].copy()

    vote_df["vote"] = 1
    nay_df["vote"] = 0
    clear_nays["vote"] = 0

    vote_df_temp = pd.concat([vote_df, nay_df, clear_nays], ignore_index=True)

    return vote_df_temp


def get_consponsor_data_for_congress(congress_num, zip_file, n_nays=10, random_type="raw"):
    '''Get the cosponsorship data for a specific Congress from the downloaded zip file

    # Args:
        congress_num (int): The congress number of interest, valid values: [93, 114]
        zip_file (ZipFile): A ZipFile object that has been loaded into memory
        n_nays (int): The number of no votes to randomly generate (all sponsors are assumed to
            be yesses)
        random_type (str): How the no votes are assigned
            - "raw" yields random selection from the set of all non-sponsors
            - "pagerank" yields a "smart" selection based on the structure of the copsonsorship network
    # Returns:
        vote_temp_df (DataFrame): A DataFrame of cosponsor "votes"
    '''
    print("Processing Congress Number {}".format(congress_num))
    # Extract the cosponsor data from the zip file
    sponsor_data = pd.read_csv(zip_file.open("govtrack_cosponsor_data_{}_congress.csv".format(congress_num)), low_memory=False)

    # Sponsor data has occasional duplicate values
    # Keep value associated with Primary sponsor
    sponsor_data = sponsor_data.sort_values(["bill_number", "sponsor"], ascending=False)
    sponsor_data = sponsor_data.drop_duplicates(["thomas_id", "bill_number"], keep='first')

    # 114th doesn't have thomas_id
    if congress_num == 114:
        sponsor_data = sponsor_data.drop("thomas_id", axis=1)
        sponsor_data = pd.merge(sponsor_data, id_crosswalk, left_on="bioguide_id", right_index=True, how="left")

    # If sponsorship withdrawn, assume that represents a clear "no" vote
    vote_df = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), ["thomas_id", "bill_number"]].copy()
    clear_nays = sponsor_data.loc[sponsor_data["date_withdrawn"].notnull(), ["thomas_id", "bill_number"]].copy()

    # Generate votes for only one chamber at a time
    # h_list and s_list are lists of legislator ids that are in the data for that legislative session
    h_list = vote_df.loc[vote_df["bill_number"].str.slice(0, 1) == "h", "thomas_id"].drop_duplicates().values
    s_list = vote_df.loc[vote_df["bill_number"].str.slice(0, 1) == "s", "thomas_id"].drop_duplicates().values

    leg_ids = sponsor_data["thomas_id"].unique()
    bill_ids = sponsor_data["bill_number"].unique()

    if random_type == "raw":
        nay_dfs = [generate_nays(bill_id, n_nays, vote_df, h_list, s_list) for bill_id in tqdm(bill_ids)]
        nay_df = pd.concat(nay_dfs)

        vote_df["vote"] = 1
        nay_df["vote"] = 0
        clear_nays["vote"] = 0

        vote_df_temp = pd.concat([vote_df, nay_df, clear_nays], ignore_index=True)
    if random_type == "pagerank":
        print('Processing Senate')
        s_df_temp = get_personalized_pageranks(sponsor_data, chamber_letter='s', n_nays=int(n_nays / 4.0),
                                               upweight_primary_sponsor=True, min_bills_sponsored=25)
        print('Processing House')
        h_df_temp = get_personalized_pageranks(sponsor_data, chamber_letter='h', n_nays=n_nays,
                                               upweight_primary_sponsor=True, min_bills_sponsored=25)
        vote_df_temp = pd.concat([s_df_temp, h_df_temp], ignore_index=True)
    return vote_df_temp


if __name__ == '__main__':
    leg_data = get_leg_info()
    leg_data.to_feather(COSPONSOR_PATH + "/us_github_leg_data.feather")
    # Generate a crosswalk between ids
    id_crosswalk = leg_data[["thomas_id", "bioguide_id"]].dropna()
    id_crosswalk["thomas_id"] = id_crosswalk["thomas_id"].astype(int)
    id_crosswalk = id_crosswalk.set_index("bioguide_id")

    # Get the cosponsor data from James Fowler's website
    request = get("http://fowler.ucsd.edu/data/govtrack_cosponsor_data.zip")
    zip_file = ZipFile(BytesIO(request.content))
    files = zip_file.namelist()

    # Data is available from 93rd to 114th congresses
    congress_range = range(93, 115)
    # random_type determines what kind of "no's" to generate
    random_types = ["raw", "pagerank"]

    # Process each individual congress and save the result
    for random_type in random_types:
        print('Generating synthetic nays using {}'.format(random_type))
        for i in congress_range:
            vote_df_temp = get_consponsor_data_for_congress(i, zip_file, n_nays=25, random_type=random_type)
            # Generate file name for the extracted cosponsorship
            file_name = COSPONSOR_PATH + "/govtrack_cosponsor_data_{}_congress_{}.feather".format(i, random_type)
            # Save individual files
            vote_df_temp.to_feather(file_name)

        print('Concatenate all the data files')
        cosponsor_data_list = []
        for i in congress_range:
            file_name = COSPONSOR_PATH + "/govtrack_cosponsor_data_{}_congress_{}.feather".format(i, random_type)
            cosponsor_data_list += [pd.read_feather(file_name)]

        # Combine all sessions into a single dataset
        vote_df = pd.concat(cosponsor_data_list, ignore_index=True)

        vote_df = vote_df.drop_duplicates(subset=["thomas_id", "bill_number"])
        vote_df = vote_df.dropna()
        vote_df["congress"] = vote_df["bill_number"].str.split("-").str.get(1).astype(int)
        vote_df["chamber"] = vote_df["bill_number"].str.slice(0, 1)

        party_data = leg_data[["thomas_id", "party", "icpsr_id"]].dropna()
        party_data["thomas_id"] = party_data["thomas_id"].astype(int)
        party_data["icpsr_id"] = party_data["icpsr_id"].astype(int)

        vote_df = pd.merge(vote_df, party_data, how="left")
        vote_df["init_value"] = 0
        vote_df.loc[vote_df["party"] == "Democrat", "init_value"] = -1
        vote_df.loc[vote_df["party"] == "Republican", "init_value"] = 1

        vote_df = vote_df.rename(columns={"thomas_id": "leg_id", "bill_number": "vote_id"})

        # vote_df["vote_id"].value_counts()
        # vote_df.sort_values("vote_id")
        # vote_df["vote"].mean()

        if random_type == "raw":
            vote_df.to_feather(COSPONSOR_PATH + "/govtrack_cosponsor_data_{}.feather".format(random_type))
        elif random_type == "pagerank":
            vote_df.to_feather(COSPONSOR_PATH + "/govtrack_cosponsor_data_{}.feather".format(random_type))

    # TODO: Upweight the things we know to be true (i.e. the yes votes)
