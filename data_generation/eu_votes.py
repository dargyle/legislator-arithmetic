'''Download EU vote data from parltrack

Parltrack compresses their files using lzip (https://www.nongnu.org/lzip/).
This needs to be installed and available via the command line.
'''

import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
# from tqdm.notebook import tqdm

from datetime import datetime

import urllib.request

from constants import DATA_PATH
EU_PATH = DATA_PATH + '/eu/'


PARL_TRACK = "https://parltrack.org/dumps/"

# To start, we will download the bulk files, save them locally at the `EU_PATH`, and
# unzip them. This only needs to be done once to have the files present, but the files
# update regularly. We focus on the files for dossiers (a rough analogue for a bill),
# Members of European Parliament (MEPs), and the parliamentary votes.

# Data acquisition
refresh_data = True
if refresh_data:
    file_names = ["ep_votes.json", "ep_meps.json"]

    # Download and unzip the files, requires lzip to be installed
    # Needs to be done prior to first run, then only when data needs to be refreshed
    for file_name in file_names:
        zip_name = file_name + '.lz'
        urllib.request.urlretrieve(PARL_TRACK + zip_name, EU_PATH + zip_name)
        # lzip -d: specifies decompress
        # lzip -f: force overwrite of existing files
        # lzip -k: keep the compressed file (default deletes the original)
        os.system('lzip -dfk ' + EU_PATH + zip_name)

# We'll begin by parsing the votes file. The code snippet includes a field mapping
# for each line of the vote data, and then various functions to process the data.
# Note that the convention of the origin files for the European
# Parliament use the following to indicate voting decisions:
# `+ = yes`, `- = no`,  and `0 = other/abstain`.

# vote data field descriptions
# - ts (timestamp): time stamp (when the vote occurred)
# - url (str): the source url
# - voteid (int): an vote specific id (originating with parltrack)
# - title (str): what is being voted on, we might call it motion in other contexts
# - votes (dict): the actual voting results
#     - '+' (dict): dictionary of information about yes votes
#         - total (int): number of yes votes
#         - groups (dict): information about yes votes
#             - %party_abbrev%: a standardized shortened party name
#                 - (list): list of votes in the form {'mepid': int},
#                     if unmatched, returns {'name': str}
#     - '-' (dict): dictionary of information about no votes (structure as above)
#     - '0' (dict): dictionary of information about abstention votes (structure as above)
# - meta (dict): metadata associated with the parltrack scraper
#     - created(timestamp): date the vote was scraped
# - changes (dict): changes to the entire dictionary
#     (active data seems to be current, so only needed for referecnce)
# - doc (str): vote document id
# - epref (list): list of epref ids

# A couple of problematic lines of the votes file: 5494, 16644, 16647. The most common problems were
# missing dictionaries of votes (mostly arising from instances where there were no entries of a
# given type) and name matching faillures that changed the structure of the vote object from the
# simple {mepid: XXXX} format to something more complicated {name: 'name', obscure_id: XXXX}.

# Debugging artifiacts for future reference
# missing votes field
# vote_entry = data_list[56]
# Name matching failures
# vote_entry = data_list[16647]
# vote_groups = vote_entry["votes"]["0"]
# These were specific examples of name matching failures referenced above
# party_block = vote_groups["groups"]["ALDE"]
# party_block = vote_groups["groups"]["S&D"]

# vote_entry = data_list[16647]
# party_abbrev = "S&D"
# vote_groups = vote_entry["votes"]["-"]
# party_block = vote_groups["groups"][party_abbrev]
# party_block


def get_roll_calls_from_list(vote_entry):
    '''This function processes the entire vote entry for a line of the parltrack vote json.

    Args:
        vote_entry (dict): a single line of the parltrack json
    Returns:
        vote_list (list): a list with all votes combined, including a vote field
    '''
    # vote_entry["votes"] is a very nested dictionary, we're going normalized it
    # First level is vote type ["+", "-", "+"] and contains a dictionary of votes indexed by party

    if "votes" in vote_entry:
        all_votes_list = []
        for vote_type, vote_party in vote_entry["votes"].items():
            # Second level is each party with a list of members voting
            for party_abbrev, mep_votes in vote_party["groups"].items():
                # Iterate over the list of voters and add fields
                for individual_votes in mep_votes:
                    individual_votes["vote_type"] = vote_type
                    individual_votes["party_abbrev"] = party_abbrev
                    individual_votes["voteid"] = vote_entry["voteid"]
                all_votes_list += mep_votes
    else:
        all_votes_list = []

    return all_votes_list


# Get the votes
# Each line is valid json, loaded here returns a list of dictionaries
with open(EU_PATH + "ep_votes.json", 'rb') as f:
    data_list = json.load(f)

# Extract the roll calls from each line into a list,
# concatenate into a single dataframe, and save.
vote_dfs = [get_roll_calls_from_list(vote_entry) for vote_entry in tqdm(data_list)]
vote_columns = ['voteid', 'party_abbrev', 'mepid', 'vote_type', 'name', 'obscure_id']
vote_df = pd.DataFrame(np.concatenate(vote_dfs).tolist(), columns=vote_columns)
vote_df.to_pickle(EU_PATH + "eu_votes.pkl")


# Now will process each line to get the voting metadata.
# Get vote metadata
def get_vote_metadata(vote_entry):
    '''Extract the vote metadata into it's own dataframe

    Args:
        vote_entry (dict): a single line of the parltrack json
    Returns:
        metadata (dict): a dictionary of the vote metadata for each line,
            processed to be more user friendly than the original
    '''

    metadata = {}
    # Get the voteid
    # Most are numeric, some are time stamps, saved as strings for simplicity
    # Note that some of the voteids are missing, which might be problematic
    try:
        metadata['voteid'] = str(vote_entry['voteid'])
    except KeyError:
        metadata['voteid'] = np.nan

    # These elements of the matdata we can pass on without any processing
    # ts: time stamp
    # url: source url for the vote data
    # title: the title of the thing being voted on
    # doc: a document reference id
    # epref: the eurpean parliament reference ids (in a list)
    var_names = ['ts', 'url', 'title', 'doc', 'epref']
    for i in var_names:
        try:
            metadata[i] = vote_entry[i]
        except KeyError:
            metadata[i] = np.nan

    # Get the vote counts for each type of vote
    try:
        metadata["yes_total"] = vote_entry["votes"]["+"]["total"]
    except KeyError:
        # If there isn't a vote for this entry, assign as missing
        metadata["yes_total"] = np.nan
    try:
        metadata["no_total"] = vote_entry["votes"]["-"]["total"]
    except KeyError:
        metadata["no_total"] = np.nan
    try:
        metadata["other_total"] = vote_entry["votes"]["0"]["total"]
    except KeyError:
        metadata["other_total"] = np.nan

    # Metadata about when the vote was created and updated
    for i in ["created", "updated"]:
        try:
            metadata[i] = vote_entry["meta"][i]
        except KeyError:
            metadata[i] = np.nan

    return metadata


# Extract the metadata and concatenate into a single DataFrame
# (assumes the datalist is already in memory from the previous section)
vote_metadata_list = [get_vote_metadata(
    vote_entry) for vote_entry in tqdm(data_list)]
vote_metadata = pd.DataFrame(vote_metadata_list)
# Convert the timestamp to a pandas timestamp
vote_metadata["ts"] = pd.to_datetime(vote_metadata["ts"])
# Save the metadata
vote_metadata.to_pickle(EU_PATH + "eu_vote_metadata.pkl")


# Now the legislator data.
with open(EU_PATH + "ep_meps.json", 'rb') as f:
    data_list = json.load(f)
meps = pd.DataFrame(data_list)


def get_people_metadata(meps_entry):
    metadata = {}

    # Basic info
    metadata["person_id"] = meps_entry["UserID"]
    metadata["active"] = meps_entry["active"]

    # Rarely present id field
    if "_id" in meps_entry:
        metadata["_id"] = meps_entry["_id"]

    # Name info
    for k, v in meps_entry["Name"].items():
        metadata["name_" + k] = v

    # Bio data
    if "Birth" in meps_entry:
        for k, v in meps_entry["Birth"].items():
            metadata["birth_" + k] = v

    if "Constituencies" in meps_entry:
        # Drop None entries from the Constituencies list
        # Some sort of data bug
        constituencies = [i for i in meps_entry["Constituencies"] if i]
        # Get the last one in list, assume most current
        metadata["home_country"] = constituencies[-1]["country"]
        metadata["home_party"] = constituencies[-1]["party"]

    # Scraper metadata
    metadata.update(meps_entry["meta"])

    # TODO: Add contact info, including address, websites, social media

    return metadata


leg_data_list = [get_people_metadata(meps_entry) for meps_entry in tqdm(data_list)]
leg_data = pd.DataFrame(leg_data_list)
leg_data["person_id"] = leg_data["person_id"].astype(str)
leg_data.to_pickle(EU_PATH + "leg_data.pkl")


def get_party_info(meps_entry):
    try:
        # Make a copy of the dictinonaries in the list
        # Avoid overwriting the original
        parties = [i.copy() for i in meps_entry["Groups"]]
    except KeyError:
        parties = []

    for party in parties:
        party["person_id"] = meps_entry["UserID"]

    return parties


party_list = [get_party_info(meps_entry) for meps_entry in tqdm(data_list)]
party_df = pd.DataFrame(np.concatenate(party_list).tolist())
party_df["start"] = pd.to_datetime(party_df["start"])
party_df["end"] = party_df["end"].str.replace('9999', '2021')
party_df["end"] = pd.to_datetime(party_df["end"])

today = datetime.now().date()
date_indexer = (party_df["start"].dt.date <= today) & (party_df["end"].dt.date >= today)
current_parties = party_df[date_indexer]

most_recent_parties = party_df.sort_values(["person_id", "end"]).drop_duplicates(subset=["person_id"], keep="last")

party_map = {"PPE-DE": "PPE",
             "PSE": "S&D",
             "ALDE": "RE",
             "UEN": "CRE",
             "ECR": "CRE",
             "ENF": "ID",
             "ITS": "ID",
             "EFD": "EFDD",
             "IND/DEM": "EFDD",
             "Group of the European United Left - Nordic Green Left": "GUE/NGL",
             "Renew": "RE"}

init_value_map = {"PPE": 0.5,
                  "S&D": -0.5,
                  "RE": 0.1,
                  "Verts/ALE": -1,
                  "CRE": 1,
                  "GUE/NGL": -1,
                  "SOC": -1,
                  }

# most_recent_parties = pd.read_pickle(EU_PATH + "most_recent_parties.pkl")
most_recent_parties["active"] = (most_recent_parties["end"].dt.date > datetime.now().date())
most_recent_parties["active"] = most_recent_parties["active"].map({False: "inactive", True: "active"})
most_recent_parties["person_id"] = most_recent_parties["person_id"].astype(str)
most_recent_parties["party_plot"] = most_recent_parties["groupid"].map(party_map)
most_recent_parties["party_plot"] = most_recent_parties["party_plot"].fillna(most_recent_parties["groupid"])
most_recent_parties["init_value"] = most_recent_parties["party_plot"].map(init_value_map)
most_recent_parties["init_value"] = most_recent_parties["init_value"].fillna(0)
# most_recent_parties["init_value"] = most_recent_parties["init_value"].astype(int)

most_recent_parties.to_pickle(EU_PATH + "most_recent_parties.pkl")
