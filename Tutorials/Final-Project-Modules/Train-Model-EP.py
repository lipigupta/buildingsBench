"""
train_model.py

Standalone script version of Train-Model.ipynb, meant to be submitted to the
Perlmutter batch scheduler (see submit_train.sh) so the full hyperparameter
sweep can run unattended overnight, instead of interactively in a notebook.

Usage:
    python train_model.py                # runs both Task 1 and Task 2 sweeps
    python train_model.py --task 1       # only the built-in NN/RNN/LSTM/GRU sweep
    python train_model.py --task 2       # only the custom MyNN sweep

Edit the two blocks marked "EDIT" below (the MyNN architecture and the sweep
configs) exactly like you would in the notebook -- everything else mirrors
the notebook's "Do Not Edit" cell verbatim.
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buildings_bench import load_torch_dataset
from buildings_bench.models import model_factory


# ------------------- #
# --- Do Not Edit --- #
# ------------------- #

class DataHandler:
    """
    Thin convenience wrapper around load_torch_dataset() + DataLoader
    construction, so Trainer doesn't need to know loading details.

    Usage:
        handler = DataHandler(batch_size=32)
        buildings = handler.load_dataset('ideal', scaler_transform='boxcox')
        loader = handler.create_dataloader(buildings[0][1])

    Args:
        batch_size (int): Batch size used by every DataLoader this creates.
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def load_dataset(self, dataset_name, scaler_transform):
        """Load a BuildingsBench dataset as a list of (building_id, dataset) pairs.

        Requires the TRANSFORM_PATH environment variable to already be set
        (path to the pickled scaler-transform data, e.g. for box-cox).

        Args:
            dataset_name (str): Dataset to load, e.g. 'ideal'.
            scaler_transform (str): '' | 'boxcox' | 'standard' -- which
                scaling transform to apply to the load values.

        Returns:
            list[tuple[str, TorchBuildingDataset]]: One entry per building.
        """
        from buildings_bench import load_torch_dataset
        return list(load_torch_dataset(
            dataset_name,
            apply_scaler_transform=scaler_transform,
            scaler_transform_path=Path(os.environ["TRANSFORM_PATH"])
        ))

    def create_dataloader(self, dataset):
        """Wrap a single building's dataset in a DataLoader.

        Args:
            dataset (TorchBuildingDataset): One building's windowed dataset.

        Returns:
            DataLoader: Batches of size self.batch_size. Not shuffled --
                order matters since these are time-series windows.
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class TimeSeriesSinusoidalPeriodicEmbedding(nn.Module):
    """
    Embeds a single periodic time feature (e.g. hour_of_day, which wraps
    23 -> 0) via sin/cos, then projects it to `embedding_dim`. Turns a
    value that wraps around into a smooth periodic representation a
    neural net can actually learn continuity from, instead of seeing a
    discontinuous jump at the wraparound point.

    Args:
        embedding_dim (int): Size of the output embedding.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x (torch.Tensor): shape (batch, seq_len, 1) -- the raw periodic value.

        Returns:
            torch.Tensor: shape (batch, seq_len, embedding_dim).
        """
        x = torch.cat([torch.sin(torch.pi * x), torch.cos(torch.pi * x)], dim=2)
        return self.linear(x)


class Model(nn.Module):
    """
    Base class for every forecasting architecture in this script (NN, RNN,
    LSTM, GRU, MyNN). Handles the plumbing shared by all of them: fixed
    context/prediction window lengths, activation-function lookup by name,
    and the per-feature input embeddings (lat/lon, building type, load,
    and the three periodic time features). Subclasses only need to
    implement `_build_model()` and `forward()`.

    Attributes:
        context_len (int): Hours of history fed to the model (168 = 1 week).
        pred_len (int): Hours to forecast (24 = 1 day).
        activation (nn.Module): Resolved activation function.
        embeddings (nn.ModuleDict): Per-feature embedding layers.
    """
    DEFAULT_CONTEXT_LEN = 168
    DEFAULT_PRED_LEN = 24

    def __init__(self, activation):
        super().__init__()
        self.context_len = self.DEFAULT_CONTEXT_LEN
        self.pred_len = self.DEFAULT_PRED_LEN
        self.activation = self._get_activation(activation)
        self.embeddings = self._create_embeddings()

    def _create_embeddings(self):
        """Build the embedding layer for each input feature.

        Returns:
            nn.ModuleDict: keys 'power', 'building', 'lat', 'lon',
                'day_of_year', 'day_of_week', 'hour_of_day'.
        """
        return nn.ModuleDict({
            'power': nn.Linear(1, 64),
            'building': nn.Embedding(2, 32),
            'lat': nn.Linear(1, 32),
            'lon': nn.Linear(1, 32),
            'day_of_year': TimeSeriesSinusoidalPeriodicEmbedding(32),
            'day_of_week': TimeSeriesSinusoidalPeriodicEmbedding(32),
            'hour_of_day': TimeSeriesSinusoidalPeriodicEmbedding(32)
        })

    def _get_activation(self, name):
        """Look up an activation module by name (case-insensitive).

        Args:
            name (str): One of 'relu', 'tanh', 'gelu', 'leaky_relu'.
                Unrecognized names silently fall back to ReLU.

        Returns:
            nn.Module: The resolved activation layer.
        """
        return {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU()
        }.get(name.lower(), nn.ReLU())

    def _data_pre_process(self, x):
        """Embed every input feature and concatenate them into one tensor.

        Args:
            x (dict[str, torch.Tensor]): Batch dict with keys 'latitude',
                'longitude', 'building_type', 'load', 'day_of_year',
                'day_of_week', 'hour_of_day'.

        Returns:
            torch.Tensor: shape (batch, seq_len, 256) -- the concatenated
                embeddings (32*6 + 64 = 256 channels), ready for a
                subclass's own layers.
        """
        lat = self.embeddings['lat'](x['latitude'])
        lon = self.embeddings['lon'](x['longitude'])
        btype = self.embeddings['building'](x['building_type'].squeeze(-1))
        load = self.embeddings['power'](x['load'])
        day_of_year = self.embeddings['day_of_year'](x['day_of_year'])
        day_of_week = self.embeddings['day_of_week'](x['day_of_week'])
        hour_of_day = self.embeddings['hour_of_day'](x['hour_of_day'])
        return torch.cat([lat, lon, btype, day_of_year, day_of_week, hour_of_day, load], dim=2)


class NN(Model):
    """
    Feedforward baseline. Flattens the entire embedded context window into
    one vector and maps it straight to the prediction window through a
    single hidden layer. Ignores sequence order entirely (unlike
    RNN/LSTM/GRU) -- useful as a lower bound for judging whether the
    recurrent models are actually learning temporal structure.
    """
    def __init__(self, activation):
        super().__init__(activation)
        self.model = self._build_model()

    def _build_model(self):
        """Build a 2-layer MLP: input_dim -> 128 -> pred_len.

        Returns:
            nn.Sequential
        """
        input_dim = self.context_len * 256
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            self.activation,
            nn.Linear(128, self.pred_len)
        )

    def forward(self, x):
        """Args:
            x (dict[str, torch.Tensor]): Batch dict, see Model._data_pre_process.

        Returns:
            torch.Tensor: shape (batch, pred_len, 1) -- the forecast.
        """
        ts_embed = self._data_pre_process(x)
        x_flat = ts_embed[:, :self.context_len, :].reshape(x['load'].shape[0], -1)
        return self.model(x_flat).unsqueeze(-1)


class RNN(Model):
    """
    Two-layer vanilla RNN. Encodes the full embedded (context+pred)
    sequence, takes the final hidden state, and maps it to the pred_len
    forecast. Simpler and more prone to vanishing gradients than LSTM/GRU
    on long sequences -- a useful comparison point for showing *why* gated
    recurrent units exist.
    """
    def __init__(self, activation="relu"):
        super().__init__(activation)
        self.rnn1, self.rnn2, self.output_layer = self._build_model()

    def _build_model(self):
        """Build a 2-layer stacked RNN (256->128->128) plus an output projection.

        Returns:
            tuple[nn.RNN, nn.RNN, nn.Linear]
        """
        rnn1 = nn.RNN(256, 128, batch_first=True)
        rnn2 = nn.RNN(128, 128, batch_first=True)
        output_layer = nn.Linear(128, self.pred_len)
        return rnn1, rnn2, output_layer

    def forward(self, x):
        """Args:
            x (dict[str, torch.Tensor]): Batch dict, see Model._data_pre_process.

        Returns:
            torch.Tensor: shape (batch, pred_len, 1) -- the forecast.
        """
        ts_embed = self._data_pre_process(x)
        out1, _ = self.rnn1(ts_embed)
        out2, _ = self.rnn2(out1)
        last_hidden = self.activation(out2[:, -1, :])
        return self.output_layer(last_hidden).unsqueeze(-1)


class LSTM(Model):
    """
    Two-layer LSTM. Same shape as RNN, but its gating (input/forget/output
    gates) lets it retain information over much longer sequences without
    vanishing gradients -- typically the strongest of the four built-in
    architectures on a 168-hour context window.
    """
    def __init__(self, activation="relu"):
        super().__init__(activation)
        self.lstm1, self.lstm2, self.output_layer = self._build_model()

    def _build_model(self):
        """Build a 2-layer stacked LSTM (256->128->128) plus an output projection.

        Returns:
            tuple[nn.LSTM, nn.LSTM, nn.Linear]
        """
        lstm1 = nn.LSTM(256, 128, batch_first=True)
        lstm2 = nn.LSTM(128, 128, batch_first=True)
        output_layer = nn.Linear(128, self.pred_len)
        return lstm1, lstm2, output_layer

    def forward(self, x):
        """Args:
            x (dict[str, torch.Tensor]): Batch dict, see Model._data_pre_process.

        Returns:
            torch.Tensor: shape (batch, pred_len, 1) -- the forecast.
        """
        ts_embed = self._data_pre_process(x)
        out1, _ = self.lstm1(ts_embed)
        out2, _ = self.lstm2(out1)
        last_hidden = self.activation(out2[:, -1, :])
        return self.output_layer(last_hidden).unsqueeze(-1)


class GRU(Model):
    """
    Two-layer GRU. A simpler gating mechanism than LSTM (no separate cell
    state, fewer parameters) that often matches LSTM's accuracy while
    training faster -- a good speed/accuracy tradeoff to compare against
    LSTM directly.
    """
    def __init__(self, activation="relu"):
        super().__init__(activation)
        self.gru1, self.gru2, self.output_layer = self._build_model()

    def _build_model(self):
        """Build a 2-layer stacked GRU (256->128->128) plus an output projection.

        Returns:
            tuple[nn.GRU, nn.GRU, nn.Linear]
        """
        gru1 = nn.GRU(256, 128, batch_first=True)
        gru2 = nn.GRU(128, 128, batch_first=True)
        output_layer = nn.Linear(128, self.pred_len)
        return gru1, gru2, output_layer

    def forward(self, x):
        """Args:
            x (dict[str, torch.Tensor]): Batch dict, see Model._data_pre_process.

        Returns:
            torch.Tensor: shape (batch, pred_len, 1) -- the forecast.
        """
        ts_embed = self._data_pre_process(x)
        out1, _ = self.gru1(ts_embed)
        out2, _ = self.gru2(out1)
        last_hidden = self.activation(out2[:, -1, :])
        return self.output_layer(last_hidden).unsqueeze(-1)


class Trainer:
    """
    Trains one (dataset, model, activation, optimizer, epochs) combination
    end-to-end and writes its results to disk.

    Owns model construction, optimizer selection, the training loop, and
    evaluation -- MAE/RMSE/R² computed in the original, unscaled load
    units via each building's `inverse_transform`. Every combination gets
    its own output directory:
        <cwd>/<dataset>/<model>/<activation>/<optimizer>/epochs-<N>/
    containing train_loss.json, predictions.json, and evaluate_model.json.

    Usage:
        trainer = Trainer(model_name='LSTM', device='cuda:0',
                           scaler_transform='boxcox', dataset_name='ideal',
                           epochs=10, train_buildings=train_buildings,
                           test_buildings=test_buildings, activation='relu',
                           optimizer_name='adam', lr=1e-3)
        train_duration = trainer.train()
        results, mae, rmse, r2 = trainer.evaluate()

    Args:
        model_name (str): One of 'NN', 'RNN', 'LSTM', 'GRU', 'MyNN'.
        device (str): 'cuda:0' or 'cpu'.
        scaler_transform (str): Scaling transform applied to the load
            values; must match what was used to build train/test_buildings
            (e.g. so `inverse_transform` in evaluate() is correct).
        dataset_name (str): Used only to build the output directory path.
        epochs (int): Number of training epochs.
        train_buildings (list[tuple[str, TorchBuildingDataset]]): Training
            split, as returned by DataHandler.load_dataset.
        test_buildings (list[tuple[str, TorchBuildingDataset]]): Held-out
            evaluation split.
        activation (str): Passed through to the model constructor.
        optimizer_name (str): One of 'adam', 'sgd', 'adamw'.
        lr (float): Learning rate.
    """
    def __init__(self, model_name, device, scaler_transform, dataset_name, epochs,
                 train_buildings, test_buildings, activation='relu',
                 optimizer_name='adam', lr=1e-3):
        self.model_name = model_name
        self.device = device
        self.scaler_transform = scaler_transform
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.train_buildings = train_buildings
        self.test_buildings = test_buildings
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.model = self._load_model()
        self.optimizer = self._get_optimizer()
        self.loss_fn = nn.MSELoss()
        self.handler = DataHandler(batch_size=32)
        self.path = os.path.join(os.getcwd(), dataset_name, model_name, activation,
                                  optimizer_name, f'epochs-{epochs}')
        os.makedirs(self.path, exist_ok=True)

    def _load_model(self):
        """Instantiate the requested architecture by name.

        Returns:
            Model: The model, moved to self.device.

        Raises:
            KeyError: If self.model_name isn't one of NN/RNN/LSTM/GRU/MyNN.
        """
        model_map = {
            'NN': NN,
            'RNN': RNN,
            'LSTM': LSTM,
            'GRU': GRU,
            'MyNN': MyNN
        }
        return model_map[self.model_name](activation=self.activation).to(self.device)

    def _get_optimizer(self):
        """Build the optimizer for self.model's parameters.

        Returns:
            torch.optim.Optimizer: Falls back to Adam if optimizer_name
                isn't recognized.
        """
        opt_map = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW
        }
        optimizer_cls = opt_map.get(self.optimizer_name.lower(), torch.optim.Adam)
        return optimizer_cls(self.model.parameters(), lr=self.lr)

    def train(self):
        """Run the full training loop for self.epochs epochs.

        Iterates every building's DataLoader each epoch, computing MSE
        loss between the model's forecast and the ground-truth load
        beyond context_len. Writes per-epoch loss plus total wall-clock
        training time to train_loss.json in self.path.

        Returns:
            float: Total training duration in seconds.
        """
        self.model.train()
        log = []
        start_time = time.time()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for building_id, building_dataset in self.train_buildings:
                dataloader = self.handler.create_dataloader(building_dataset)
                for batch in dataloader:
                    for key, value in batch.items():
                        batch[key] = value.to(self.device)
                    self.optimizer.zero_grad()
                    predictions = self.model(batch)
                    targets = batch['load'][:, self.model.context_len:, 0]
                    loss = self.loss_fn(predictions[:, :, 0], targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
            print(f"[{self.model_name}] Epoch {epoch + 1}: Loss = {total_loss:.4f}", flush=True)
            log.append({"epoch": epoch + 1, "loss": total_loss})
        train_duration = time.time() - start_time
        with open(os.path.join(self.path, "train_loss.json"), "w") as f:
            json.dump({"train_loss": log, "train_duration": train_duration}, f, indent=2)
        return train_duration

    def evaluate(self):
        """Evaluate the trained model on self.test_buildings.

        Runs inference with no gradient tracking, inverse-transforms
        predictions/targets/loads back to real kWh units per building
        (undoing whatever scaler_transform was applied at load time), and
        averages MAE/RMSE/R² across buildings. Writes raw predictions to
        predictions.json and the averaged metrics to evaluate_model.json.

        Returns:
            tuple[dict, float, float, float]: (per-building results dict
                with 'load'/'predictions'/'targets' lists, mean MAE, mean
                RMSE, mean R²).
        """
        self.model.eval()
        results = {}
        mae_total = 0.0
        rmse_total = 0.0
        r2_total = 0.0
        count = 0
        for building_id, building_dataset in self.test_buildings:
            inverse_transform = building_dataset.datasets[0].load_transform.undo_transform
            dataloader = self.handler.create_dataloader(building_dataset)

            target_list = []
            prediction_list = []
            load_list = []

            with torch.no_grad():
                for batch in dataloader:
                    for key, value in batch.items():
                        batch[key] = value.to(self.device)

                    predictions = self.model(batch)
                    targets = batch['load'][:, self.model.context_len:]
                    loads = batch['load'][:, :self.model.context_len]

                    targets = inverse_transform(targets)
                    predictions = inverse_transform(predictions)
                    loads = inverse_transform(loads)

                    prediction_list.append(predictions.detach().cpu())
                    target_list.append(targets.detach().cpu())
                    load_list.append(loads.detach().cpu())

            predictions_all = torch.cat(prediction_list)
            targets_all = torch.cat(target_list)
            load_all = torch.cat(load_list)

            mae = torch.abs(predictions_all - targets_all).mean().item()
            rmse = torch.sqrt(((predictions_all - targets_all) ** 2).mean()).item()
            r2 = 1 - (((predictions_all - targets_all) ** 2).sum() /
                      ((targets_all - targets_all.mean()) ** 2).sum()).item()
            mae_total += mae
            rmse_total += rmse
            r2_total += r2
            count += 1
            results[building_id] = {
                "load": load_all.tolist(),
                "predictions": predictions_all.tolist(),
                "targets": targets_all.tolist()
            }
        with open(os.path.join(self.path, "predictions.json"), "w") as f:
            json.dump(results, f, indent=2)
        eval_metrics = {
            "mae": mae_total / count,
            "rmse": rmse_total / count,
            "r2": r2_total / count}
        with open(os.path.join(self.path, "evaluate_model.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)
        return results, eval_metrics["mae"], eval_metrics["rmse"], eval_metrics["r2"]


# ------------------- #
# ------ EDIT ------- #
# ------------------- #
# Task 2: build your custom architecture here (same exercise as the notebook's Task 2)

class MyNN(Model):
    """
    Task 2: your custom architecture. Reuses Model's input embedding
    pipeline (flattened context window -> context_len*256 vector) --
    fill in _build_model() with your own nn.Sequential stack.
    """
    def __init__(self, activation):
        super().__init__(activation)
        self.model = self._build_model()

    def _build_model(self):
        """TODO: build your own nn.Sequential.

        Needs, in order: an input layer sized (input_dim -> some hidden
        width), at least three hidden layers, and an output layer sized
        (-> self.pred_len).

        Returns:
            nn.Sequential
        """
        input_dim = self.context_len * 256
        return nn.Sequential(
            nn.Linear(input_dim, 256),     # input layer
            self.activation,
            nn.Linear(256, 128),           # hidden layer 1
            self.activation,
            nn.Linear(128, 64),            # hidden layer 2
            self.activation,
            nn.Linear(64, 32),             # hidden layer 3
            self.activation,
            nn.Linear(32, self.pred_len)   # output layer -- must end at pred_len
        )

    def forward(self, x):
        """Args:
            x (dict[str, torch.Tensor]): Batch dict, see Model._data_pre_process.

        Returns:
            torch.Tensor: shape (batch, pred_len, 1) -- the forecast.
        """
        ts_embed = self._data_pre_process(x)
        x_flat = ts_embed[:, :self.context_len, :].reshape(x['load'].shape[0], -1)
        return self.model(x_flat).unsqueeze(-1)


# ------------------- #
# ------ EDIT ------- #
# ------------------- #
# Sweep configuration. Edit these lists exactly like you would the notebook's
# Edit cells -- this is what actually runs overnight, so this is where you'd
# put the FULL 30+/10+ combo sweep rather than the notebook's quick-debug subset.

TASK1_CONFIG = {
    "dataset_names": ["ideal"],          # TODO: your assigned dataset(s), e.g. ["ideal"]
    "model_classes": ["NN", "RNN", "LSTM", "GRU"],
    "activations": ["relu", "tanh", "leaky_relu", "gelu"],
    "optimizers": ["adam", "sgd", "adamw"],
    "epoch_options": [1],
}

TASK2_CONFIG = {
    "dataset_names": ["ideal"],          # TODO: your assigned dataset(s)
    "model_classes": ["MyNN"],
    "activations": ["relu", "tanh", "leaky_relu", "gelu"],
    "optimizers": ["adam", "sgd", "adamw"],
    "epoch_options": [1],
}

# ------------------- #
# --- Do Not Edit --- #
# ------------------- #


def flatten_combos(config):
    """Turn a TASK*_CONFIG dict into a flat list of individual run specs.

    This is what makes distributing the sweep across GPUs/nodes possible --
    once it's one flat list, "which combos does worker N handle" is just
    list slicing, regardless of how many dimensions the original sweep had.

    Args:
        config (dict): Must provide 'dataset_names', 'model_classes',
            'activations', 'optimizers', 'epoch_options', each a list.

    Returns:
        list[dict]: One dict per combo, keys 'dataset_name', 'model_class',
            'activation', 'optimizer_name', 'epochs'.
    """
    combos = []
    for dataset_name in config["dataset_names"]:
        for model_class in config["model_classes"]:
            for activation in config["activations"]:
                for optimizer_name in config["optimizers"]:
                    for epochs in config["epoch_options"]:
                        combos.append({
                            "dataset_name": dataset_name,
                            "model_class": model_class,
                            "activation": activation,
                            "optimizer_name": optimizer_name,
                            "epochs": epochs,
                        })
    return combos


# Cache of (train_buildings, test_buildings) per dataset name, so a single
# worker process handling multiple combos on the same dataset only loads
# and splits it once instead of once per combo.
_dataset_cache = {}


def get_dataset_split(dataset_name):
    """Load a dataset and its 80/20 train/test building split, cached per process.

    Args:
        dataset_name (str): Dataset to load, e.g. 'ideal'.

    Returns:
        tuple[list, list]: (train_buildings, test_buildings)
    """
    if dataset_name not in _dataset_cache:
        print(f"\n=== Loading dataset: {dataset_name} ===", flush=True)
        handler = DataHandler(batch_size=32)
        all_buildings = handler.load_dataset(dataset_name, scaler_transform="boxcox")
        train_buildings = all_buildings[:int(0.8 * len(all_buildings))]
        test_buildings = all_buildings[int(0.8 * len(all_buildings)):]
        _dataset_cache[dataset_name] = (train_buildings, test_buildings)
    return _dataset_cache[dataset_name]


def run_combo(combo, device, worker_label=""):
    """Train + evaluate a single combo dict (one output of flatten_combos()).

    Args:
        combo (dict): One entry from flatten_combos().
        device (str): 'cuda:0' or 'cpu'.
        worker_label (str): Optional prefix for log lines, e.g. "[task 3/16]",
            so overlapping parallel workers' output stays distinguishable.
    """
    train_buildings, test_buildings = get_dataset_split(combo["dataset_name"])
    print(f"\n{worker_label} --- Training {combo['model_class']} | "
          f"Activation: {combo['activation']} | Optimizer: {combo['optimizer_name']} | "
          f"Epochs: {combo['epochs']} | Dataset: {combo['dataset_name']} ---", flush=True)
    trainer = Trainer(
        model_name=combo["model_class"],
        device=device,
        dataset_name=combo["dataset_name"],
        epochs=combo["epochs"],
        train_buildings=train_buildings,
        test_buildings=test_buildings,
        scaler_transform="boxcox",
        activation=combo["activation"],
        optimizer_name=combo["optimizer_name"],
        lr=1e-3,
    )
    train_duration = trainer.train()
    results, mae, rmse, r2 = trainer.evaluate()
    print(f"{worker_label} [{combo['model_class']}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
          f"R²: {r2:.4f}, Training Time: {train_duration:.2f}s", flush=True)


def run_sweep(config, device, task_id=0, num_tasks=1):
    """Train and evaluate this worker's slice of `config`'s cross-product.

    Flattens the sweep into one combo list, then takes a round-robin slice
    (combos[task_id::num_tasks]) so that N cooperating workers (one per
    GPU, e.g. launched via `srun` with --gpus-per-task=1) collectively
    cover every combo exactly once with no coordination beyond knowing
    their own rank. With the defaults (task_id=0, num_tasks=1) this covers
    every combo in a single process, same as running serially.

    Args:
        config (dict): See flatten_combos().
        device (str): 'cuda:0' or 'cpu'.
        task_id (int): This worker's 0-indexed rank among num_tasks.
        num_tasks (int): Total number of parallel workers splitting the sweep.
    """
    combos = flatten_combos(config)
    my_combos = combos[task_id::num_tasks]
    label = f"[task {task_id}/{num_tasks}]" if num_tasks > 1 else ""
    print(f"{label} {len(my_combos)} of {len(combos)} total combos assigned to this worker", flush=True)
    for combo in my_combos:
        run_combo(combo, device, worker_label=label)


def write_timing_record(run_tag, task_id, num_tasks, start_epoch, end_epoch):
    """Write this task's wall-clock start/end to timing/<run_tag>/task_<id>.json.

    Uses time.time() (absolute Unix epoch seconds), not a relative "elapsed
    seconds" counter, so records from *different processes* -- possibly on
    different nodes -- can later be compared directly: the earliest start
    across every task and the latest end across every task give the whole
    job's true wall-clock makespan, with no shared coordination between
    tasks needed (each just writes its own file; see compute_speedup.py).
    This only assumes node clocks are synchronized, which is standard on
    an HPC cluster like Perlmutter.

    Args:
        run_tag (str): Groups files from one job/run together, e.g. 'serial'
            vs 'parallel16', so separate experiments don't overwrite each
            other's timing files.
        task_id (int): This worker's rank.
        num_tasks (int): Total workers in this run.
        start_epoch (float): time.time() when this task started working.
        end_epoch (float): time.time() when this task finished.
    """
    timing_dir = os.path.join(os.getcwd(), "timing", run_tag)
    os.makedirs(timing_dir, exist_ok=True)
    record = {
        "task_id": task_id,
        "num_tasks": num_tasks,
        "start": start_epoch,
        "end": end_epoch,
        "duration": end_epoch - start_epoch,
    }
    with open(os.path.join(timing_dir, f"task_{task_id}.json"), "w") as f:
        json.dump(record, f, indent=2)


def main():
    """Parse args and run this worker's slice of the requested sweep(s).

    Also sets the BUILDINGS_BENCH/TRANSFORM_PATH/REPO_PATH environment
    variable defaults (only if not already set by the calling shell/sbatch
    script) and picks cuda:0 if a GPU is available, else cpu.

    Distributing across GPUs/nodes: this script doesn't need mpi4py or
    torch.distributed -- each combo is a fully independent training run,
    so "parallel" here just means "N copies of this same script, each
    told which slice of the combo list is theirs." --task-id/--num-tasks
    default to $SLURM_PROCID/$SLURM_NTASKS, which `srun` sets automatically
    for every task it launches -- see submit_train_parallel.sh, which
    launches one task per GPU across the whole node/GPU allocation with a
    single `srun python train_model.py`. Because --gpus-per-task=1 is set
    there, each task's `cuda:0` already refers to its own assigned
    physical GPU -- no per-task GPU-selection code is needed here.

    Timing: every run also writes a timing/<run_tag>/task_<id>.json file
    (see write_timing_record) so you can measure real wall-clock speedup
    between a serial run and a parallel run without touching `sacct` --
    just run compute_speedup.py against the two run-tags' directories
    afterward.
    """
    job_start_epoch = time.time()

    parser = argparse.ArgumentParser(description="Batch sweep runner for Train-Model.ipynb")
    parser.add_argument("--task", choices=["1", "2", "both"], default="both",
                         help="Which sweep to run: 1 (built-in models), 2 (MyNN), or both")
    parser.add_argument("--task-id", type=int, default=None,
                         help="This worker's 0-indexed rank among --num-tasks. "
                              "Defaults to $SLURM_PROCID (0 if unset).")
    parser.add_argument("--num-tasks", type=int, default=None,
                         help="Total number of parallel workers splitting the sweep. "
                              "Defaults to $SLURM_NTASKS (1 if unset).")
    parser.add_argument("--run-tag", type=str, default="default",
                         help="Label for this run's timing files, e.g. 'serial' or "
                              "'parallel16' -- keeps separate experiments' timing "
                              "records from overwriting each other.")
    args = parser.parse_args()

    task_id = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_PROCID", 0))
    num_tasks = args.num_tasks if args.num_tasks is not None else int(os.environ.get("SLURM_NTASKS", 1))

    os.environ.setdefault("REPO_PATH", "/global/cfs/cdirs/m4388/Project4/BuildingsBench")
    os.environ.setdefault("BUILDINGS_BENCH", "/global/cfs/cdirs/m4388/Project4/Dataset")
    os.environ.setdefault("TRANSFORM_PATH", "/global/cfs/cdirs/m4388/Project4/Dataset/metadata/transforms")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[task {task_id}/{num_tasks}] Using device: {device}", flush=True)

    if args.task in ("1", "both"):
        print(f"\n#### TASK 1: Predefined models (worker {task_id}/{num_tasks}) ####", flush=True)
        run_sweep(TASK1_CONFIG, device, task_id, num_tasks)

    if args.task in ("2", "both"):
        print(f"\n#### TASK 2: Custom MyNN (worker {task_id}/{num_tasks}) ####", flush=True)
        run_sweep(TASK2_CONFIG, device, task_id, num_tasks)

    job_end_epoch = time.time()
    print(f"[task {task_id}/{num_tasks}] Total wall time for this worker's combos: "
          f"{job_end_epoch - job_start_epoch:.2f}s", flush=True)
    write_timing_record(args.run_tag, task_id, num_tasks, job_start_epoch, job_end_epoch)


if __name__ == "__main__":
    main()