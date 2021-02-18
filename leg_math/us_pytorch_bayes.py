import pyro
import pyro.distributions as dist
import torch

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD
import pyro.contrib.autoguide as autoguides

from pyro.infer.autoguide.initialization import init_to_value

import pandas as pd
import numpy as np
import pickle
import os

from data_generation.data_processing import process_data, format_model_data

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

from leg_math.pytorch_bayes import bayes_irt_basic, bayes_irt_full, normalize_ideal_points

from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s',
                    level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler("debug1.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

pyro.enable_validation(True)

# Set up environment
gpu = True

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

US_PATH = DATA_PATH + '/us/'
if not os.path.exists(US_PATH):
    os.makedirs(US_PATH)

logger.info("Read in and process all the US votes")
vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
vote_df["time_vote"] = vote_df["congress"]
# Limit to one chamber for now
# vote_df = vote_df[vote_df["chamber"] == "Senate"]
# vote_df = vote_df[vote_df["congress"] == 74]

# k_dim = 2
# k_time = 0
# covariates_list = []
vb_metrics = []
mcmc_metrics = []
for k_dim in range(1, 4):
    for k_time in range(0, 2):
        for covariates_list in [[], ["in_majority"]]:
            data_params = dict(
                           congress_cutoff=0,
                           k_dim=k_dim,
                           k_time=k_time,
                           covariates_list=covariates_list,
                           unanimity_check=True,
                           )
            vote_data = process_data(vote_df=vote_df, **data_params)
            custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

            x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

            # Convert training and test data to tensors
            legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
            votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
            responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
            if covariates_list:
                covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
            if k_time > 0:
                time_tensor = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)
                time_present = torch.tensor(vote_data["time_present"], device=device)    
                
            legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
            votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
            responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
            if covariates_list:
                covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)            
            if k_time > 0:
                time_tensor_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)
                
            # Set some constants
            n_legs = vote_data["J"]
            n_votes = vote_data["M"]
            if covariates_list:
                n_covar = covariates.shape[1]
                
            # Make train and test data bundles
            core_data = [legs, n_legs, votes, n_votes]
            core_data_test = [legs_test, n_legs, votes_test, n_votes]
            aux_data = {}
            aux_data_test = {}            
            if covariates_list:
                aux_data["k_dim"] = k_dim
                aux_data["covariates"] = covariates
                aux_data["n_covar"] = n_covar
                
                aux_data_test["k_dim"] = k_dim                
                aux_data_test["covariates"] = covariates_test
                aux_data_test["n_covar"] = n_covar                
            if k_time > 0:
                aux_data["k_time"] = k_time
                aux_data["time_passed"] = time_tensor
                
                aux_data_test["k_time"] = k_time
                aux_data_test["time_passed"] = time_tensor_test                 

            # Setup the optimizer
            optim = Adam({'lr': 0.1})

            # Define the guide
            guide = autoguides.AutoNormal(bayes_irt_full)
            # guide = autoguides.AutoNormal(bayes_irt_basic, init_loc_fn=init_to_value(values={'theta': custom_init_values}))

            # Setup the variational inference
            svi = SVI(bayes_irt_full, guide, optim, loss=Trace_ELBO())

            logger.info("Fit the model using variational bayes")
            # Run variational inference
            pyro.clear_param_store()
            min_loss = float('inf') # initialize to infinity
            patience = 0
            for j in range(5000):
                loss = svi.step(*core_data, y=responses, device=device, **aux_data)                
                if j % 100 == 0:
                    logger.info("[epoch %04d] loss: %.4f" % (j + 1, loss))
                    min_loss = min(loss, min_loss)
                    if (loss > min_loss):
                        if patience >= 4:
                            break
                        else:
                            patience += 1
                    else:
                        patience = 0


            # Save the parameters for future use
            pyro.get_param_store().save(US_PATH + f'bayes/params_vb_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl')        

            # # Get the parameters into pandas dataframes
            # ideal_points = pd.concat(
            #                 [pd.DataFrame(pyro.param("AutoNormal.locs.theta").cpu().data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
            #                  pd.DataFrame(pyro.param("AutoNormal.scales.theta").cpu().data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
            #                  ], axis=1)
            # polarity = pd.concat(
            #                 [pd.DataFrame(pyro.param("AutoNormal.locs.beta").cpu().data.numpy(), columns=['loc_{}'.format(j + 1) for j in range(k_dim)]),
            #                  pd.DataFrame(pyro.param("AutoNormal.scales.beta").cpu().data.numpy(), columns=['scale_{}'.format(j + 1) for j in range(k_dim)])
            #                  ], axis=1)
            # popularity = pd.DataFrame({"loc": pyro.param('AutoNormal.locs.alpha').cpu().data.numpy(), "scale": pyro.param('AutoNormal.scales.alpha').cpu().data.numpy()})

            logger.info("Generate predictions for model evaluation")
            n_samples = 1000

            # This is the "normal" case, our models are large enough we exhaust memory, hence the iterative approach
            # posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, guide=guide, num_samples=n_samples, return_sites=["obs"])
            # preds = posterior_predictive(legs, votes, covariates=covariates, device=device)["obs"].mean(axis=0).squeeze()
            # preds_test = posterior_predictive(legs_test, votes_test, covariates=covariates_test, device=device)["obs"].mean(axis=0).squeeze()

            posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, guide=guide, num_samples=1, return_sites=["obs"])

            preds = torch.zeros(size=responses.shape, device=device)
            preds_test = torch.zeros(size=responses_test.shape, device=device)
            for _ in tqdm(range(n_samples)):
                preds += posterior_predictive(*core_data, device=device, **aux_data)["obs"].squeeze()
                preds_test += posterior_predictive(*core_data_test, device=device, **aux_data_test)["obs"].squeeze()
            preds = preds / n_samples
            preds_test = preds_test / n_samples

            # Define metrics
            criterion = torch.nn.BCELoss(reduction="mean")
            log_like = torch.nn.BCELoss(reduction="sum")
            k = 0
            for param_name, param_value in pyro.get_param_store().items():
                if "locs" in param_name:
                    k += np.array(param_value.shape).prod()

            # Calculate log loss and accuracy scores, in sample
            train_metrics = {"bce": criterion(preds, responses).item(),
                             "log_like": log_like(preds, responses).item(),
                             "accuracy": (1.0 * (responses == preds.round())).mean().item(), 
                            }
            logger.info(f'Train loss (VB): {train_metrics["bce"]}')
            logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

            # Calculate log loss and accuracy scores, out of sample
            test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                            "log_like": log_like(preds_test, responses_test).item(),
                            "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(), 
                            }
            logger.info(f'Test loss (VB): {test_metrics["bce"]}')
            logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

            train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
            test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

            metrics = {**train_metrics, **test_metrics}

            metrics["train_n"] = responses.shape[0]
            metrics["train_k"] = k
            metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
            metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

            metrics["test_n"] = responses_test.shape[0]
            metrics["test_k"] = k
            metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
            metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])        

            final_metrics = {**data_params, **metrics}
            pd.Series(final_metrics).to_pickle(US_PATH + f'bayes/metrics_vb_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl')
            print(pd.Series(final_metrics))
            vb_metrics += [final_metrics]

            #############
            # MCMC Time #
            #############

            logger.info("Now fit the model using sampling")
            # Initialize the MCMC to our estimates from the variational model
            # Speeds things up and requires less burn-in
            if covariates_list:
                init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                                    "beta": pyro.param("AutoNormal.locs.beta").data,
                                                    "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                                    "coef": pyro.param("AutoNormal.locs.coef").data,
                                                    })
            else:
                init_values = init_to_value(values={"theta": pyro.param("AutoNormal.locs.theta").data,
                                                    "beta": pyro.param("AutoNormal.locs.beta").data,
                                                    "alpha": pyro.param("AutoNormal.locs.alpha").data,
                                                    })    
                
            # Set up sampling alrgorithm
            nuts_kernel = NUTS(bayes_irt_full, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True, init_strategy=init_values)
            # For real inference should probably increase the number of samples, but this is part is slow and enough to test
            hmc_posterior = MCMC(nuts_kernel, num_samples=250, warmup_steps=100)
            # Run the model
            hmc_posterior.run(*core_data, y=responses, device=device, **aux_data)

            # Save the posterior
            hmc_posterior.sampler = None
            hmc_posterior.kernel.potential_fn = None
            with open(US_PATH + f'bayes/params_mcmc_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl', 'wb') as f:
                pickle.dump(hmc_posterior, f)
            # with open(US_PATH + f'bayes/params_mcmc_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl', 'rb') as f:
            #     hmc_posterior = pickle.load(f)  

            # Again, this would be the normal way, can't do it because of memory issues
            # samples = hmc_posterior.get_samples()
            # # Generate model predictions based on the posterior samples
            # posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, samples)
            # preds = posterior_predictive(legs, votes, covariates=covariates, device=device)["obs"].mean(axis=0).flatten(0)
            # preds_test = posterior_predictive(legs_test, votes_test, covariates=covariates_test, device=device)["obs"].mean(axis=0).flatten(0)

            preds = torch.zeros(size=responses.shape, device=device)
            preds_test = torch.zeros(size=responses_test.shape, device=device)
            for _ in tqdm(range(n_samples)):
                temp_samples = hmc_posterior.get_samples(num_samples=1)
                posterior_predictive = pyro.infer.predictive.Predictive(bayes_irt_full, posterior_samples=temp_samples, return_sites=["obs"])
                preds += posterior_predictive(*core_data, device=device, **aux_data)["obs"].squeeze()
                preds_test += posterior_predictive(*core_data_test, device=device, **aux_data_test)["obs"].squeeze()
            preds = preds / n_samples
            preds_test = preds_test / n_samples

            # Calculate log loss and accuracy scores, in sample
            train_metrics = {"bce": criterion(preds, responses).item(),
                             "log_like": log_like(preds, responses).item(),
                             "accuracy": (1.0 * (responses == preds.round())).mean().item(), 
                            }
            logger.info(f'Train loss (VB): {train_metrics["bce"]}')
            logger.info(f'Train accuracy (VB): {train_metrics["accuracy"]}')

            # Calculate log loss and accuracy scores, out of sample
            test_metrics = {"bce": criterion(preds_test, responses_test).item(),
                            "log_like": log_like(preds_test, responses_test).item(),
                            "accuracy": (1.0 * (responses_test == preds_test.round())).mean().item(), 
                            }
            logger.info(f'Test loss (VB): {test_metrics["bce"]}')
            logger.info(f'Test accuracy (VB): {test_metrics["accuracy"]}')

            train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
            test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

            metrics = {**train_metrics, **test_metrics}

            metrics["train_n"] = responses.shape[0]
            metrics["train_k"] = k
            metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
            metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])

            metrics["test_n"] = responses_test.shape[0]
            metrics["test_k"] = k
            metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
            metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])        

            final_metrics = {**data_params, **metrics}
            pd.Series(final_metrics).to_pickle(US_PATH + f'bayes/metrics_mcmc_{k_dim}_{k_time}_{"".join(covariates_list)}.pkl')
            print(pd.Series(final_metrics))
            mcmc_metrics += [final_metrics]

metrics_df = pd.DataFrame(vb_metrics)
metrics_df.to_pickle(US_PATH + f'bayes/all_metrics_vb.pkl')            
            
metrics_df = pd.DataFrame(mcmc_metrics)
metrics_df.to_pickle(US_PATH + f'bayes/all_metrics_mcmc.pkl')

# ideal_points_mcmc = pd.concat([
#                         pd.DataFrame(samples["theta"].mean(axis=0).cpu().numpy(), columns=['loc_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
#                         pd.DataFrame(samples["theta"].std(axis=0).cpu().numpy(), columns=['scale_{}_mcmc'.format(j + 1) for j in range(k_dim)]),
#                         ], axis=1)

# # sns.distplot(pd.DataFrame(samples["coef"].cpu().numpy()))

# # Compare thre results of the two processes
# comp_ideal = pd.concat([ideal_points, ideal_points_mcmc], axis=1)
# comp_ideal.describe()
# comp_ideal.corr()
# # comp_ideal.plot(kind='scatter', x="loc_1", y="loc_1_mcmc")
# # comp_ideal.plot(kind='scatter', x="scale_1_mcmc", y="scale_1")

# import seaborn as sns
# sns.distplot(pd.Series(samples["theta"][:, 0, 0].cpu().numpy()), bins=25)

# temp_d = dist.Normal(ideal_points.loc[0, "loc_1"], ideal_points.loc[0, "scale_1"])
# sns.distplot(pd.Series([temp_d().item() for k in range(100)]), bins=25)


# transformed_params_mcmc = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"])
# transformed_params_mcmc = normalize_ideal_points(samples["theta"], samples["beta"], samples["alpha"], verify_predictions=True)
# transformed_params = normalize_ideal_points(pyro.param("AutoNormal.locs.theta").data.unsqueeze(0),
#                                             pyro.param("AutoNormal.locs.beta").data.unsqueeze(0),
#                                             pyro.param("AutoNormal.locs.alpha").data.unsqueeze(0))


# from pyro.ops.stats import hpdi, waic
# hpdi(samples["coef"], prob=0.95)
# waic(hmc_posterior)
#
#
# # samples = hmc_posterior.get_samples(group_by_chain=False)
# # predictive = pyro.infer.Predictive(self.model, samples)
# # vectorized_trace = predictive.get_vectorized_trace(*self._args, **self._kwargs)
# # for obs_name in self.observations.keys():
# #     obs_site = vectorized_trace.nodes[obs_name]
# #     log_like = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
# #     shape = (self.nchains, self.ndraws) + log_like.shape[1:]
# #     data[obs_name] = np.reshape(log_like, shape)
#
# posterior_predictive(legs, votes, covariates=covariates)
# vectorized_trace = posterior_predictive.get_vectorized_trace(legs, votes, covariates=covariates)
#
#
# from pyro.infer.predictive import _predictive
# _predictive(self.model, posterior_samples, self.num_samples, return_trace=True, model_args=args, model_kwargs=kwargs)
# hmc_posterior.get_samples()["coef"]
# torch.mm(covariates, hmc_posterior.get_samples()["coef"])
# _predictive(bayes_irt_full, hmc_posterior.get_samples(), 250, return_trace=True, model_args=[legs, votes], model_kwargs={"covariates": covariates})
#
#
#
#
#
# vote_df = pd.read_feather(DATA_PATH + "vote_df_cleaned.feather")
# vote_df = vote_df[vote_df["chamber"] == "Senate"]
#
# # first_session = vote_df.groupby("leg_id")[["congress"]].agg(["min", "max"])
# # first_session.columns = ["first_session", "last_session"]
# # first_session["sessions_served"] = first_session["last_session"] - first_session["first_session"]
# # # first_session["first_session"].value_counts()
# # vote_df = pd.merge(vote_df, first_session, left_on="leg_id", right_index=True)
# # vote_df["time_passed"] = vote_df["congress"] - vote_df["first_session"]
#
# # pd.DataFrame(np.stack([(vote_df["time_passed"] ** i).values for i in range(0, k_time + 1)]).transpose())
#
# k_dim = 2
# k_time = 2
# data_params = dict(
#                vote_df=vote_df,
#                congress_cutoff=110,
#                k_dim=k_dim,
#                k_time=k_time,
#                covariates_list=[],
#                unanimity_check=False,
#                )
# vote_data = process_data(**data_params)
# custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)
#
# x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)
#
# # Convert training and test data to tensors
# legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
# votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
# responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
# # covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
# time_passed = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)
#
# # pd.DataFrame(np.stack(vote_data["time_passed_train"]).transpose()).sort_values(1)
#
# legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
# votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
# responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
# # covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
# time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)
#
# # time_tensor = torch.cat([torch.ones(time_passed.shape[0], device=device).unsqueeze(-1), time_passed], axis=1)
# time_tensor = time_passed
# # time_tensor_test = torch.cat([torch.ones(time_passed_test.shape[0], device=device).unsqueeze(-1), time_passed_test], axis=1)
# time_tensor_test = time_passed_test
#
# sessions_served = torch.tensor(vote_data["sessions_served"])
#
# # Set some constants
# n_legs = len(set(legs.numpy()))
# n_votes = len(set(votes.numpy()))
# # n_covar = covariates.shape[1]
#
# wnom_model = wnom_full(n_legs, n_votes, k_dim, pretrained=custom_init_values, k_time=data_params["k_time"])
#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)
#
# # w = wnom_model.ideal_points[torch.arange(0, 100)]
#
# losses = []
# accuracies = []
# test_losses = []
# test_accuracies = []
# for t in tqdm(range(5000)):
#     y_pred = wnom_model(legs, votes, time_tensor, sessions_served)
#     loss = criterion(y_pred, responses)
#
#     with torch.no_grad():
#         # accuracy = accuracy_score(responses, y_pred.numpy() >= 0.5)
#         accuracy = ((y_pred > 0) == responses).sum().item() / len(responses)
#
#         y_pred_test = wnom_model(legs_test, votes_test, time_tensor_test)
#         loss_test = criterion(y_pred_test, responses_test)
#         # accuracy_test = accuracy_score(responses_test, y_pred_test.numpy() >= 0.5)
#         accuracy_test = ((y_pred_test > 0) == responses_test).sum().item() / len(responses_test)
#
#     if t % 100 == 99:
#         print(t, loss.item())
#
#     losses.append(loss.item())
#     accuracies.append(accuracy)
#     test_losses.append(loss_test.item())
#     test_accuracies.append(accuracy_test)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # print(loss)
#
# wnom_model.ideal_points[torch.arange(0, n_legs)]
# time_tensor.unsqueeze(-1).shape
# wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1)
# torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1)
# torch.sum(wnom_model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=1).norm(dim=1)
#
# wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)
#
# pd.DataFrame(wnom_model.ideal_points[torch.arange(0, n_legs)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
# pd.DataFrame(wnom_model.ideal_points[torch.arange(0, n_legs), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
# wnom_model.ideal_points[torch.arange(0, n_legs)].sum(dim=2).norm(dim=1).max()
#
# wnom_model.beta
# wnom_model.w
#
# wnom_model.ideal_points.shape
# initial_ideal = wnom_model.ideal_points[torch.arange(0, n_legs), :, 0]
# initial_ideal.shape
# last_session_ideal = initial_ideal + wnom_model.ideal_points[torch.arange(0, n_legs), :, 1] * sessions_served.unsqueeze(-1) + wnom_model.ideal_points[torch.arange(0, n_legs), :, 2] * (sessions_served ** 2).unsqueeze(-1)
#
# ideal_points = wnom_model.ideal_points[torch.arange(0, n_legs), :, :]
# asdf = torch.clone(initial_ideal)
# for kk in range(1, ideal_points.shape[2]):
#     asdf += ideal_points[:, : , kk] * sessions_served.pow(kk).unsqueeze(-1)
# asdf
#
# last_session_ideal.shape
# pd.Series(last_session_ideal.norm(dim=1).detach().numpy()).hist()
# asdf.norm(dim=1).max()
# pd.DataFrame(asdf.detach().numpy()).plot(kind="scatter", x=0, y=1)
