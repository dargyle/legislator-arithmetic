import os
import pyro
import torch

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

import pandas as pd
import numpy as np

from data_generation.data_processing import process_data, format_model_data

from constants import DATA_PATH

from sklearn.metrics import accuracy_score

from leg_math.pytorch_wnom import wnom_basic, wnom_full, gen_shuffled_data

from tqdm import tqdm

from qhoptim.pyt import QHM, QHAdam

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

pyro.enable_validation(True)

# Set up environment
gpu = True

if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

EU_PATH = DATA_PATH + '/eu/'
if not os.path.exists(EU_PATH):
    os.makedirs(EU_PATH)

logger.info("Load and process the EU data")
most_recent_parties = pd.read_feather(EU_PATH + "most_recent_parties.feather")

eu_votes = pd.read_feather(EU_PATH + "eu_votes.feather")
eu_votes["voteid"] = eu_votes["voteid"].astype(str)

vote_metadata = pd.read_feather(EU_PATH + "eu_vote_metadata.feather")
vote_metadata["voteid"] = vote_metadata["voteid"].astype(str)
# vote_metadata['time_passed'] = vote_metadata['ts'].dt.year - 2004
vote_metadata["congress"] = vote_metadata['ts'].dt.year
vote_metadata["time_vote"] = vote_metadata['ts'].dt.year
vote_time_passed = vote_metadata[["voteid", "congress", "time_vote"]]
eu_votes = pd.merge(eu_votes, vote_time_passed, on='voteid')

vote_df = eu_votes.rename(columns={"voteid": "vote_id"})
vote_df = vote_df.drop(columns=["name", "obscure_id"])
vote_df = vote_df.dropna(subset=["mepid"]).copy()
vote_df["leg_id"] = vote_df["mepid"].astype(int).astype(str)
vote_df["vote"] = vote_df["vote_type"].map({"+": 1, "-": 0, "0": np.nan})
vote_df = vote_df.dropna().copy()
vote_df["vote"] = vote_df["vote"].astype(int)

vote_df = pd.merge(vote_df,
                   most_recent_parties[["person_id", "init_value"]],
                   left_on='leg_id',
                   right_on="person_id",
                   how='left')

all_metrics = []
# k_dim = 1
# k_time = 1
for k_dim in range(1, 7):
    for k_time in range(0, 2):
        logger.info(f"Processing for k_dim={k_dim} and k_time={k_time}")
        data_params = dict(
                       congress_cutoff=0,
                       k_dim=k_dim,
                       k_time=k_time,
                       covariates_list=[],
                       unanimity_check=False,
                       )
        vote_data = process_data(vote_df=vote_df, **data_params)
        custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

        x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

        # Convert training and test data to tensors
        legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
        votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
        responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
        if k_time > 0:
            time_tensor = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)
            time_present = torch.tensor(vote_data["time_present"], device=device)

        legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
        votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
        responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
        if k_time > 0:
            time_tensor_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

        # Set some constants
        n_legs = torch.unique(legs).shape[0]
        n_votes = torch.unique(votes).shape[0]

        # Adjust init values for time setup
        if k_time > 0:
            first_dim = custom_init_values
            other_dims = 0.000001 * (torch.rand(n_legs, k_dim, k_time, device=device) - 0.5)
            pretrained = torch.cat([first_dim.unsqueeze(-1), other_dims], dim=2)
        else:
            pretrained = custom_init_values
        
        
        # Old reliable way to fit the model, works fine but missing some features
        # logger.info("Set up the pytorch model")
        #
        # wnom_model = wnom_full(n_legs, n_votes, k_dim, pretrained).to(device)
        #
        # criterion = torch.nn.BCEWithLogitsLoss()
        # # optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)
        # optimizer = QHAdam(wnom_model.parameters(), lr=1e-2)
        #
        # logger.info("Fit the pytorch model")
        # losses = []
        # accuracies = []
        # test_losses = []
        # test_accuracies = []
        # best_loss = np.inf
        # keep_best = False
        # for t in tqdm(range(5000)):
        #     y_pred = wnom_model(legs, votes)
        #     loss = criterion(y_pred, responses)
        #
        #     with torch.no_grad():
        #         accuracy = ((y_pred > 0) == responses).sum().item() / len(responses)
        #
        #         y_pred_test = wnom_model(legs_test, votes_test)
        #         loss_test = criterion(y_pred_test, responses_test)
        #         accuracy_test = ((y_pred_test > 0) == responses_test).sum().item() / len(responses_test)
        #
        #         if keep_best:
        #             if loss_test < best_loss:
        #                 best_loss = loss_test
        #                 torch.save(wnom_model.state_dict(), 'best-model-parameters.pt')
        #
        #     # if t % 100 == 0:
        #     #     logger.info(f'epoch {t}, loss: {loss.item()}')
        #
        #     losses.append(loss.item())
        #     accuracies.append(accuracy)
        #     test_losses.append(loss_test.item())
        #     test_accuracies.append(accuracy_test)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()


        logger.info("Set up a pytorch model with ignite")
        if k_time > 0:
            model = wnom_full(n_legs, n_votes, k_dim, pretrained, k_time=k_time).to(device)
        else:
            model = wnom_full(n_legs, n_votes, k_dim, pretrained).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        log_like = torch.nn.BCEWithLogitsLoss(reduction="sum")
        # optimizer = torch.optim.AdamW(wnom_model.parameters(), amsgrad=True)
        # Default learning rate is too conservative, this works well for this dataset
        optimizer = QHAdam(model.parameters(), lr=5e-2)

        # Data loader was super slow, but probably the "right" way to do this
        # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(legs, votes, responses), batch_size=100000, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(legs_test, votes_test, responses_test), batch_size=100000)

        # Instead, just make an iterable of a list of the data
        # In theory, this could be a list of separate batches, but in our case using the whole dataset is fine       
        n_shuffle = 10
        if k_time > 0:
            train_shuffle_loader = gen_shuffled_data(1000 * n_shuffle, [legs, votes, responses, time_tensor], [time_present], p=0.8, shuffle=True)
            train_loader = [[legs, votes, responses, time_tensor] + [time_present]]
            val_loader = [[legs_test, votes_test, responses_test, time_tensor_test] + [time_present]]
        else:
            train_shuffle_loader = gen_shuffled_data(1000 * n_shuffle, [legs, votes, responses])
            train_loader = [[legs, votes, responses]]
            val_loader = [[legs_test, votes_test, responses_test]]


        def process_function(engine, batch):
            model.train()
            optimizer.zero_grad()
            if model.k_time > 0:
                x1, x2, y, tt, ss = batch
                y_pred = model(x1, x2, tt, ss)
            else:
                x1, x2, y = batch
                y_pred = model(x1, x2)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            return y_pred, y


        def eval_function(engine, batch):
            model.eval()
            with torch.no_grad():
                if model.k_time > 0:
                    x1, x2, y, tt, ss = batch
                    y_pred = model(x1, x2, tt, ss)
                else:
                    x1, x2, y = batch
                    y_pred = model(x1, x2)
                return y_pred, y


        trainer = Engine(process_function)
        train_evaluator = Engine(eval_function)
        validation_evaluator = Engine(eval_function)

        # RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(torch.sigmoid(y_pred))
            return y_pred, y


        Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
        Loss(criterion).attach(train_evaluator, 'bce')
        Loss(log_like).attach(train_evaluator, 'log_like')

        Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')
        Loss(criterion).attach(validation_evaluator, 'bce')
        Loss(log_like).attach(validation_evaluator, 'log_like')
        
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer)
        # pbar.attach(trainer, ['loss'])

        def score_function(engine):
            # logger.warning(engine.state.metrics)
            val_loss = engine.state.metrics['bce']
            return -val_loss

        handler = EarlyStopping(patience=9, score_function=score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, handler)

        checkpointer = ModelCheckpoint(EU_PATH + "checkpoints", f'shuffle_{k_dim}_{k_time}', n_saved=None, create_dir=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'wnom_full': model})

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_bce = metrics['bce']
            log_like = -1 * metrics['log_like']
            pbar.log_message(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}  Log-likelihood= {:.1f}"
                .format(engine.state.epoch, avg_accuracy, avg_bce, log_like))


        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            validation_evaluator.run(val_loader)
            metrics = validation_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_bce = metrics['bce']
            log_like = -1 * metrics['log_like']
            pbar.log_message(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}  Log-likelihood= {:.1f}"
                .format(engine.state.epoch, avg_accuracy, avg_bce, log_like))
            pbar.n = pbar.last_print_n = 0

        trainer.run(train_shuffle_loader, max_epochs=1000, epoch_length=n_shuffle) 
            
        logger.info("Generate some accuracy metrics after the model is done")
        y_pred, _ = eval_function(trainer, train_loader[0])
        train_metrics = train_evaluator.run(train_loader).metrics
        train_metrics = {'train_' + k: v for k, v, in train_metrics.items()}
        test_metrics = validation_evaluator.run(val_loader).metrics
        test_metrics = {'test_' + k: v for k, v, in test_metrics.items()}

        # Deprecated due to machine percision issues
        log_like_old = (responses * torch.log(torch.sigmoid(y_pred)) + (1 - responses) * torch.log(1 - torch.sigmoid(y_pred))).sum()
        
        k = 0
        for param in model.parameters():
            # print(param.shape)
            k += np.array(param.shape).prod()

        metrics = {**train_metrics, **test_metrics}
        metrics["log_like_old"] = log_like_old.item()
    
        metrics["train_n"] = responses.shape[0]
        metrics["train_k"] = k
        metrics["train_aic"] = ((2 * k) - (2 * -1 * metrics["train_log_like"]))
        metrics["train_bic"] = k * np.log(metrics["train_n"]) - (2 * -1 * metrics["train_log_like"])
        
        metrics["test_n"] = responses_test.shape[0]
        metrics["test_k"] = k
        metrics["test_aic"] = ((2 * k) - (2 * -1 * metrics["test_log_like"]))
        metrics["test_bic"] = k * np.log(metrics["test_n"]) - (2 * -1 * metrics["test_log_like"])        
        
        final_metrics = {**data_params, **metrics}
        pd.Series(final_metrics).to_pickle(EU_PATH + f'checkpoints/metrics_shuffle_wnom_full_{k_dim}_{k_time}.pkl')
        print(pd.Series(final_metrics))
        all_metrics += [final_metrics]

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_pickle(EU_PATH + f'checkpoints/all_metrics_shuffle_wnom_full.pkl')

metrics_df = pd.read_pickle(EU_PATH + f'checkpoints/all_metrics_shuffle_wnom_full.pkl')
print(metrics_df.transpose())

# m = torch.nn.Dropout(p=0.2)
# input = torch.randn(5, 4)
# output = m(input)

# m = torch.nn.Dropout2d(p=0.5)
# m(input.unsqueeze(-1).unsqueeze(-1)).shape

# torch.sum(model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=2).max()
# torch.sum(model.ideal_points[legs] * time_tensor.unsqueeze(1), axis=2).min()
# model.ideal_points[:, 0, 0]

# asdf = torch.sum(model.ideal_points * torch.stack([torch.ones(size=time_present.shape), time_present], dim=1).unsqueeze(1), axis=2)
# asdf
# k_dim

# model = wnom_full(n_legs, n_votes, k_dim, pretrained, k_time=k_time)
# model.load_state_dict(torch.load(EU_PATH + "checkpoints/" + f'wnom_full_{k_dim}_{k_time}_wnom_full_199.pt'))
# temp_ideal_1 = model.ideal_points.detach().numpy()

# model = wnom_full(n_legs, n_votes, k_dim, pretrained, k_time=k_time)
# model.load_state_dict(torch.load(EU_PATH + "checkpoints/" + f'wnom_full_{k_dim}_{k_time}_wnom_full_198.pt'))
# temp_ideal_2 = model.ideal_points.detach().numpy()

# (temp_ideal_1 - temp_ideal_2)

# metrics_df["n"] = responses.shape[0]
# metrics_df["bic"] = metrics_df["k"] * np.log(metrics_df["n"]) - (2 * metrics_df["log_like"])
# np.exp((metrics_df["aic"].min() - metrics_df["aic"]) / 2)
# (metrics_df["bic"].min() - metrics_df["bic"])

# pd.Series(losses).plot()
# pd.Series(test_losses).plot()
#
# pd.Series(accuracies).plot()
# pd.Series(test_accuracies).plot()
#
#
#
# true_ideal = random_votes[["coord1D", "coord2D"]]
# true_ideal.index = true_ideal.index.droplevel("vote_id")
# true_ideal = true_ideal.drop_duplicates()
# vote_data.keys()
# leg_crosswalk_rev = {v: k for k, v in vote_data["leg_crosswalk"].items()}
# true_ideal[["wnom1", "wnom2"]] = wnom_model.ideal_points[torch.tensor(true_ideal.index.map(leg_crosswalk_rev).values)].detach().numpy()
#
# true_ideal.corr()
# true_ideal.plot(kind='scatter', x="coord1D", y="wnom1")
# true_ideal.plot(kind='scatter', x="coord2D", y="wnom2")
#
# X = wnom_model.ideal_points[torch.arange(0, 100)].detach()
# Y = torch.tensor(true_ideal[["coord1D", "coord2D"]].values, dtype=torch.float)
#
# ab = torch.inverse(X.transpose(0, 1).mm(X))
# cd = X.transpose(0, 1).mm(Y)
# rot = ab.mm(cd)
#
# from scipy.linalg import orthogonal_procrustes
# rot, _ = orthogonal_procrustes(X, Y)
# temp_X = X.mm(torch.tensor(rot))
#
# true_ideal[["wnom1", "wnom2"]] = temp_X.numpy()
# true_ideal.corr()
#
# pd.DataFrame(temp_X.numpy()).plot(kind='scatter', x=0, y=1)
#
# (wnom_model.yes_points[torch.arange(0, 100)] ** 2).sum(dim=1).max()
# pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].detach().numpy()).plot(kind='scatter', x=0, y=1)
# wnom_model.w
# wnom_model.beta
# pd.Series(y_pred.detach().numpy()).hist()
#
#
# # For the time version
# wnom_model.ideal_points[torch.arange(0, 100)]
# time_tensor.unsqueeze(-1).shape
# wnom_model.ideal_points[legs] * time_tensor[votes]
# torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2)
# torch.sum(wnom_model.ideal_points[legs] * time_tensor[votes], axis=2).norm(dim=1)
#
# wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).norm(2, dim=1)
#
# pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100)].sum(dim=2).detach().numpy()).plot(kind="scatter", x=0, y=1)
# pd.DataFrame(wnom_model.ideal_points[torch.arange(0, 100), :, 0].detach().numpy()).plot(kind="scatter", x=0, y=1)
