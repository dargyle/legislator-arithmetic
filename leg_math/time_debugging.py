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

from leg_math.pytorch_wnom import wnom_basic, wnom_full

from tqdm import tqdm

from qhoptim.pyt import QHM, QHAdam

from data_generation.random_votes import generate_nominate_votes

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)

pyro.enable_validation(True)


gpu = False
if gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

random_votes = generate_nominate_votes(n_leg=50, n_votes=1000, beta=10.0, beta_covar=0.0, k_dim=1, k_time=1, w=np.array([1.0]), cdf_type="logit", drop_unanimous_votes=False, replication_seed=42)
vote_df = random_votes.reset_index()

k_dim = 1
k_time = 1
data_params = dict(
               vote_df=vote_df,
               congress_cutoff=110,
               k_dim=k_dim,
               k_time=k_time,
               covariates_list=[],
               unanimity_check=False,
               )
vote_data = process_data(**data_params)
custom_init_values = torch.tensor(vote_data["init_embedding"].values, dtype=torch.float, device=device)

x_train, x_test, sample_weights = format_model_data(vote_data, data_params, weight_by_frequency=False)

# Convert training and test data to tensors
legs = torch.tensor(x_train[0].flatten(), dtype=torch.long, device=device)
votes = torch.tensor(x_train[1].flatten(), dtype=torch.long, device=device)
responses = torch.tensor(vote_data["y_train"].flatten(), dtype=torch.float, device=device)
# covariates = torch.tensor(vote_data["covariates_train"], dtype=torch.float, device=device)
time_passed = torch.tensor(np.stack(vote_data["time_passed_train"]).transpose(), dtype=torch.float, device=device)

legs_test = torch.tensor(x_test[0].flatten(), dtype=torch.long, device=device)
votes_test = torch.tensor(x_test[1].flatten(), dtype=torch.long, device=device)
responses_test = torch.tensor(vote_data["y_test"].flatten(), dtype=torch.float, device=device)
# covariates_test = torch.tensor(vote_data["covariates_test"], dtype=torch.float, device=device)
time_passed_test = torch.tensor(np.stack(vote_data["time_passed_test"]).transpose(), dtype=torch.float, device=device)

# time_tensor = torch.cat([torch.ones(time_passed.shape[0], device=device).unsqueeze(-1), time_passed], axis=1)
time_tensor = time_passed
# time_tensor_test = torch.cat([torch.ones(time_passed_test.shape[0], device=device).unsqueeze(-1), time_passed_test], axis=1)
time_tensor_test = time_passed_test

vote_data.keys()
time_present = torch.tensor(vote_data["time_present"], device=device)

# Set some constants
n_legs = torch.unique(legs).shape[0]
n_votes = torch.unique(votes).shape[0]


logger.info("Set up a pytorch model with ignite")
k_time = 1
if k_time > 0:
    model = wnom_full(n_legs, n_votes, k_dim, k_time=k_time).to(device)
else:
    model = wnom_full(n_legs, n_votes, k_dim, custom_init_values).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
# Default learning rate is too conservative, this works well for this dataset
optimizer = QHAdam(model.parameters(), weight_decay=1e-1)

# Data loader was super slow, but probably the "right" way to do this
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(legs, votes, responses), batch_size=100000, shuffle=True)
# val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(legs_test, votes_test, responses_test), batch_size=100000)

# Instead, just make an iterable of a list of the data
# In theory, this could be a list of separate batches, but in our case using the whole dataset is fine
if k_time > 0:
    train_loader = [[legs, votes, responses, time_tensor, time_present]]
    val_loader = [[legs_test, votes_test, responses_test, time_tensor_test, time_present]]
else:
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

Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

pbar = ProgressBar(persist=True)
pbar.attach(trainer)
# pbar.attach(trainer, ['loss'])


def score_function(engine):
    # logger.warning(engine.state.metrics)
    val_loss = engine.state.metrics['bce']
    return -val_loss


handler = EarlyStopping(patience=50, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)

checkpointer = ModelCheckpoint('.', f'wnom_full_{k_dim}_{k_time}', n_saved=9, create_dir=True, require_empty=False)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'wnom_full': model})

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    validation_evaluator.run(val_loader)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))
    pbar.n = pbar.last_print_n = 0


trainer.run(train_loader, max_epochs=1000)



vote_df.sort_values(["leg_id", "vote_id"])
vote_df["coord1D"].min()
ideal_points = vote_df.reset_index()[["leg_id", "coord1D", "coord1D_t1"]].drop_duplicates(keep="first").set_index("leg_id")
yes_points = vote_df.reset_index()[["vote_id", "yes_coord1D"]].drop_duplicates().set_index("vote_id")
no_points = vote_df.reset_index()[["vote_id", "no_coord1D"]].drop_duplicates().set_index("vote_id")


model.ideal_points[legs].shape
time_tensor.unsqueeze(1).shape
ideal_points_use = torch.sum((model.ideal_points[legs] * time_tensor.unsqueeze(1)), axis=2)
asdf = pd.DataFrame({"leg_id": legs, "vote_id": votes, "d1": ideal_points_use.squeeze().detach().numpy()})
asdf["leg_id"] = asdf["leg_id"].map(vote_data["leg_crosswalk"])
asdf["vote_id"] = asdf["vote_id"].map(vote_data["vote_crosswalk"])

asdf[asdf["leg_id"] == "leg_01"].sort_values("vote_id").set_index("vote_id").plot()

zxcv = pd.merge(vote_df.reset_index()[["leg_id", "vote_id", "coord1D"]], asdf)
zxcv.plot(kind="scatter", x="coord1D", y="d1")
zxcv[zxcv["leg_id"] == "leg_21"].plot(kind="scatter", x="coord1D", y="d1")

wnom_ideal_points = pd.DataFrame(model.ideal_points.squeeze().detach().numpy(), columns=["d1", "dt1"])
wnom_ideal_points.index = wnom_ideal_points.index.map(vote_data["leg_crosswalk"])
wnom_yes_points = pd.DataFrame(model.yes_points.detach().numpy(), columns=["d1"])
wnom_yes_points.index = wnom_yes_points.index.map(vote_data["vote_crosswalk"])
wnom_no_points = pd.DataFrame(model.no_points.detach().numpy(), columns=["d1"])
wnom_no_points.index = wnom_no_points.index.map(vote_data["vote_crosswalk"])


wnom_yes_points.sort_index().plot()

pd.merge(ideal_points, wnom_ideal_points, left_index=True, right_index=True).corr()
pd.merge(yes_points, wnom_yes_points, left_index=True, right_index=True).plot(kind="scatter", x="yes_coord1D", y="d1")
pd.merge(no_points, wnom_no_points, left_index=True, right_index=True).corr()
