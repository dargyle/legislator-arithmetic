
import os
import numpy as np
import pandas as pd

from requests import get
from io import BytesIO
from zipfile import ZipFile

import igraph

DATA_PATH = os.path.expanduser("~/data/leg_math/") + "cosponsor/"


# congress_num = 105
# bill_id = 'sconres110-105'
def generate_nays(bill_id, n_nays, leg_ids, vote_df, h_list, s_list):
    # print(bill_id)
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


def get_leg_info():
    current_leg = pd.read_csv("https://theunitedstates.io/congress-legislators/legislators-current.csv")
    historical_leg = pd.read_csv("https://theunitedstates.io/congress-legislators/legislators-historical.csv")
    leg_data = pd.concat([historical_leg, current_leg], ignore_index=True)
    leg_data.to_feather(DATA_PATH + "/us_github_leg_data.feather")
    return leg_data


def get_personalized_pageranks(sponsor_data, chamber_letter='s', n_nays=10, upweight_primary_sponsor=True, min_bills_sponsored=25):
    bill_sponsors = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), ["thomas_id", "bill_number", "sponsor"]].copy()

    bill_sponsors = bill_sponsors[bill_sponsors["bill_number"].str.slice(0, 1) == chamber_letter]

    n_bill_sponsors = bill_sponsors["thomas_id"].value_counts()
    valid_leg_ids = n_bill_sponsors[n_bill_sponsors > min_bills_sponsored].index.tolist()
    bill_sponsors = bill_sponsors[bill_sponsors["thomas_id"].isin(valid_leg_ids)]


    if upweight_primary_sponsor:
        # This divides "credit" evenly among sponsors and the set of copsonsors
        # This specific setup also implcitly downweights bills w/o cosponsors
        link_weights = 1.0 / bill_sponsors.groupby(["bill_number", "sponsor"])[["thomas_id"]].count()
        link_weights.columns = ["link_weights"]
        bill_sponsors = pd.merge(bill_sponsors, link_weights, left_on=["bill_number", "sponsor"], right_index=True)
        # Ensure that solo sponsored bills have full weight
        sponsor_type_counts = link_weights.index.get_level_values("bill_number").value_counts().to_frame(name="sponsor_type_counts")
        bill_sponsors = pd.merge(bill_sponsors, sponsor_type_counts, left_on="bill_number", right_index=True)
        bill_sponsors["link_weights"] = bill_sponsors["link_weights"] / bill_sponsors["sponsor_type_counts"]
    else:
        link_weights = 1.0 / bill_sponsors.groupby("bill_number")[["leg_id"]].count()
        link_weights.columns = ["link_weights"]
        bill_sponsors = pd.merge(bill_sponsors, link_weights, left_on="bill_number", right_index=True)

    assert np.allclose(bill_sponsors.groupby("bill_number")["link_weights"].sum().values, 1.0), "bill weights don't sum to 1"

    g_bill_sponsor = igraph.Graph.TupleList(bill_sponsors[["bill_number", "thomas_id", "link_weights"]].values.tolist(), weights=True)
    # overall_centrality = pd.DataFrame(g_bill_sponsor.personalized_pagerank(weights="weight", damping=0.5),
    #                                   index=g_bill_sponsor.vs["name"], columns=["centrality"])
    # leg_centrality = overall_centrality[~overall_centrality.index.isin(bill_sponsors["bill_number"])]
    bill_ids = bill_sponsors["bill_number"].unique()

    inferred_vote_list = []
    for bill_id in bill_ids:
        print(bill_id)
        specific_sponsors = bill_sponsors[bill_sponsors["bill_number"] == bill_id].set_index("thomas_id")
        reset_values = specific_sponsors.loc[g_bill_sponsor.vs["name"], "link_weights"].fillna(0).values
        assert np.allclose(reset_values.sum(), 1.0), "invalid reset distribution"
        bill_centrality = pd.DataFrame(g_bill_sponsor.personalized_pagerank(weights="weight", damping=0.5, reset=reset_values.tolist()),
                                       index=g_bill_sponsor.vs["name"], columns=["centrality"])
        bill_centrality = bill_centrality[~bill_centrality.index.isin(bill_sponsors["bill_number"])]

        if bill_centrality.size > 0:
            n_possible_nays = len(bill_centrality)
            use_inverse = True

            if use_inverse:
                inverse_centrality = 1 / bill_centrality
                inverse_centrality = inverse_centrality / inverse_centrality.sum()
                inverse_centrality.sort_values("centrality", ascending=False)
                nays_ids = inverse_centrality.sample(n=min(n_nays, n_possible_nays), weights="centrality", replace=False).index.tolist()
            else:
                bill_centrality = bill_centrality / bill_centrality.sum()
                bill_centrality.sort_values("centrality", ascending=False)
                nays_ids = bill_centrality.sample(n=min(n_nays, n_possible_nays), weights="centrality", replace=False).index.tolist()
            result_df = pd.DataFrame(nays_ids, columns=["thomas_id"])
            result_df["bill_number"] = bill_id
        else:
            result_df = pd.DataFrame()
        inferred_vote_list += [result_df]

    nay_df = pd.concat(inferred_vote_list)
    vote_df = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), ["thomas_id", "bill_number"]].copy()
    clear_nays = sponsor_data.loc[sponsor_data["date_withdrawn"].notnull(), ["thomas_id", "bill_number"]].copy()

    vote_df["vote"] = 1
    nay_df["vote"] = 0
    clear_nays["vote"] = 0

    vote_df_temp = pd.concat([vote_df, nay_df, clear_nays], ignore_index=True)

    return vote_df_temp


def get_consponsor_data_for_congress(congress_num, zip_file, n_nays=10, random_type="raw"):
    print("Processing Congress Number {}".format(congress_num))
    if random_type == "raw":
        file_name = DATA_PATH + "/govtrack_cosponsor_data_{}_congress.feather".format(congress_num)
    if random_type == "cosponsor":
        file_name = DATA_PATH + "/govtrack_cosponsor_data_{}_congress_smart_oppose.feather".format(congress_num)

    try:
        vote_df_temp = pd.read_feather(file_name)
    except:
        sponsor_data = pd.read_csv(zip_file.open("govtrack_cosponsor_data_{}_congress.csv".format(congress_num)))
        if congress_num == 114:
            # 114th doesn't have thomas_id
            sponsor_data = sponsor_data.drop("thomas_id", axis=1)
            sponsor_data = pd.merge(sponsor_data, id_crosswalk, left_on="bioguide_id", right_index=True, how="left")

        # If sponsorship withdrawn, clear no?
        vote_df = sponsor_data.loc[sponsor_data["date_withdrawn"].isnull(), ["thomas_id", "bill_number"]].copy()
        clear_nays = sponsor_data.loc[sponsor_data["date_withdrawn"].notnull(), ["thomas_id", "bill_number"]].copy()

        h_list = vote_df.loc[vote_df["bill_number"].str.slice(0, 1) == "h", "thomas_id"].drop_duplicates().values
        s_list = vote_df.loc[vote_df["bill_number"].str.slice(0, 1) == "s", "thomas_id"].drop_duplicates().values

        leg_ids = sponsor_data["thomas_id"].unique()
        bill_ids = sponsor_data["bill_number"].unique()

        if random_type == "raw":
            nay_dfs = [generate_nays(bill_id, n_nays, leg_ids, vote_df, h_list, s_list) for bill_id in bill_ids]
            nay_df = pd.concat(nay_dfs)

            vote_df["vote"] = 1
            nay_df["vote"] = 0
            clear_nays["vote"] = 0

            vote_df_temp = pd.concat([vote_df, nay_df, clear_nays], ignore_index=True)
        if random_type == "cosponsor":
            s_df_temp = get_personalized_pageranks(sponsor_data, chamber_letter='s', n_nays=int(n_nays / 4.0),
                                                   upweight_primary_sponsor=True, min_bills_sponsored=25)
            h_df_temp = get_personalized_pageranks(sponsor_data, chamber_letter='h', n_nays=n_nays,
                                                   upweight_primary_sponsor=True, min_bills_sponsored=25)
            vote_df_temp = pd.concat([s_df_temp, h_df_temp], ignore_index=True)
        vote_df_temp.to_feather(file_name)
    return vote_df_temp


leg_data = get_leg_info()
id_crosswalk = leg_data[["thomas_id", "bioguide_id"]].dropna()
id_crosswalk["thomas_id"] = id_crosswalk["thomas_id"].astype(int)
id_crosswalk = id_crosswalk.set_index("bioguide_id")

# https://codenhagen.wordpress.com/2015/08/21/how-to-download-and-unzip-a-file-with-python/
request = get("http://fowler.ucsd.edu/data/govtrack_cosponsor_data.zip")
zip_file = ZipFile(BytesIO(request.content))
files = zip_file.namelist()

random_type = "cosponsor"
if random_type == "raw":
    cosponsor_data_list = [get_consponsor_data_for_congress(i, zip_file, n_nays=25, random_type="raw") for i in range(93, 115)]
elif random_type == "cosponsor":
    cosponsor_data_list = [get_consponsor_data_for_congress(i, zip_file, n_nays=40, random_type="cosponsor") for i in range(93, 115)]
vote_df = pd.concat(cosponsor_data_list, ignore_index=True)

vote_df = vote_df.drop_duplicates(subset=["thomas_id", "bill_number"])
vote_df = vote_df.dropna()
vote_df["congress"] = vote_df["bill_number"].str.split("-").str.get(1).astype(int)
vote_df["chamber"] = vote_df["bill_number"].str.slice(0, 1)

vote_df = pd.merge(vote_df, leg_data[["thomas_id", "party", "icpsr_id"]])
vote_df["thomas_id"] = vote_df["thomas_id"].astype(int)
# vote_df["icpsr_id"] = vote_df["icpsr_id"].astype(int)
vote_df["init_value"] = 0
vote_df.loc[vote_df["party"] == "Democrat", "init_value"] = -1
vote_df.loc[vote_df["party"] == "Republican", "init_value"] = 1

vote_df = vote_df.rename(columns={"thomas_id": "leg_id", "bill_number": "vote_id"})

vote_df["vote_id"].value_counts()
vote_df.sort_values("vote_id")
vote_df["vote"].mean()

if random_type == "raw":
    vote_df.to_feather(DATA_PATH + "/govtrack_cosponsor_data.feather")
elif random_type == "cosponsor":
    vote_df.to_feather(DATA_PATH + "/govtrack_cosponsor_data_smart_oppose.feather")
