'''
given set of observations, fit irt model and write outputs to disk
subject: train_noise
item: imageID
y: response
'''

import argparse
import csv

import numpy as np
import pyro
import torch
import torch.nn as nn

from py_irt.models.one_param_logistic import OneParamLog
from scipy.special import expit

import pandas as pd
from data_generation.data_processing import process_data, format_model_data

from constants import DATA_PATH

# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--num-epochs', default=1000, type=int)
# parser.add_argument('--gpu', action='store_true')
# parser.add_argument('--priors', help='[vague, hierarchical]', default='hierarchical')
# parser.add_argument('--model', help='[1PL,2PL]', default='1PL')  # 2pl not implemented yet
# parser.add_argument('--response-patterns', help='file with response pattern data', required=True)
# args = parser.parse_args()



gpu = False

device = torch.device('cpu')
if gpu:
    device = torch.device('cuda')

vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df = vote_df[vote_df["chamber"] == "House"]

i = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=116,
               k_dim=i,
               k_time=1,
               covariates_list=[],
               )
vote_data = process_data(**data_params)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

# models = []
# items = []
# responses = []

# itemID2idx = {}
# idx2itemID = {}
# modelID2idx = {}
# idx2modelID = {}
#
# m_idx = 0
# i_idx = 0
#
# with open(args.response_patterns, 'r') as infile:
#     inreader = csv.reader(infile, delimiter='\t')
#     for mID, iID, response in inreader:
#         if mID not in modelID2idx:
#             modelID2idx[mID] = m_idx
#             idx2modelID[m_idx] = mID
#             m_idx += 1
#         if iID not in itemID2idx:
#             itemID2idx[iID] = i_idx
#             idx2itemID[i_idx] = iID
#             i_idx += 1
#         models.append(modelID2idx[mID])
#         items.append(itemID2idx[iID])
#         responses.append(eval(response))

models = x_train[0].flatten()
items = x_train[1].flatten()
responses = vote_data["y_train"].flatten()

num_models = len(set(models))
num_items = len(set(items))
print(num_items, num_models)

models = torch.tensor(models, dtype=torch.long, device=device)
items = torch.tensor(items, dtype=torch.long, device=device)
responses = torch.tensor(responses, dtype=torch.float, device=device)

# 3. define model and guide accordingly
model = '1PL'
priors = 'hierarchical'
num_epochs = 250
if model == '1PL':
    m = OneParamLog(priors, device, num_items, num_models)

# 4. fit irt model with svi, trace-elbo loss
m.fit(models, items, responses, num_epochs)

for name in pyro.get_param_store().get_all_param_names():
    print(name)
    if gpu:
        val = pyro.param(name).data.cpu().numpy()
    else:
        val = pyro.param(name).data.numpy()
    print(val)

# m.fit_MCMC(models, items, responses, num_epochs)

from pyro.infer.mcmc import MCMC, NUTS
from py_irt.models.one_param_logistic import OneParamLog

nuts_kernel = NUTS(m.model_vague, adapt_step_size=True)
hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100, num_chains=2)
hmc_posterior.run(models, items, responses)

hmc_posterior.summary()

type(hmc_posterior)
theta_sum = m.summary(hmc_posterior, ['theta']).items()
b_sum = m.summary(hmc_posterior, ['b']).items()

# 5. once model is fit, write outputs (diffs and thetas) to disk,
#       retaining original modelIDs and itemIDs so we can use them

pyro.get_param_store().get_all_param_names()


for name in pyro.get_param_store().get_all_param_names():
    print(name)
    if gpu:
        val = pyro.param(name).data.cpu().numpy()
    else:
        val = pyro.param(name).data.numpy()
    print(val)
    # if name == 'loc_diff':
    #     with open(args.response_patterns + '.diffs', 'w') as outfile:
    #         outwriter = csv.writer(outfile, delimiter=',')
    #         for i in range(len(val)):
    #             row = [idx2itemID[i], val[i]]
    #             outwriter.writerow(row)
    # elif name == 'loc_ability':
    #     with open(args.response_patterns + '.theta', 'w') as outfile:
    #         outwriter = csv.writer(outfile, delimiter=',')
    #         for i in range(len(val)):
    #             row = [idx2modelID[i], val[i]]
    #             outwriter.writerow(row)
