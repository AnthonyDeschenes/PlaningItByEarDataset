import os

import numpy as np
import poutyne
import torch
import torch.nn.functional as F
from joblib import dump, load
from poutyne import Model as PoutyneModel, EarlyStopping, BestModelRestore
from torch.optim import AdamW


class Model:
    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def score(self, x, y):
        pass


class ModelTrainer(Model):
    def __init__(self, num_input=1001, num_input_channels=64, batch_size=16, num_per_hidden=256,
                 lr=1e-5, latent_size=2,
                 reduce_lr_patience=15, stop_patience=30,
                 network_class=None, loss=F.mse_loss, verbose=True, beta=1.0, name="ae-1000", use_attention=True):
        super().__init__()
        batch_size_to_use = 256
        cuda_device = 0
        self.input_size = num_input
        self.use_attention = use_attention
        self.num_input_channels = num_input_channels
        self.name = name
        self.num_per_hidden = num_per_hidden
        self.device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
        if batch_size is not None:
            batch_size_to_use = batch_size
        self.batch_size = batch_size_to_use
        self.random_state = np.random.RandomState(42)
        self.model = None
        self.loss = loss
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        self.stop_patience = stop_patience
        self.network_class = network_class
        self.verbose = verbose
        self.beta = beta
        self.latent_size = latent_size
        self.max_value = 9999.0
        self.min_value = 9999.0

    def load_model(self, name, default_model):
        model = default_model
        training_needed = True
        if os.path.exists(os.path.join(os.path.dirname(__file__), f'{name}.joblib')):
            model = load(os.path.join(os.path.dirname(__file__), f'{name}.joblib'))
            training_needed = False
        return model, training_needed

    def save_model(self, name, model):
        dump(model, os.path.join(os.path.dirname(__file__), f'{name}.joblib'))

    def init_model(self):
        network = self.network_class(self.input_size,
                                     input_num_chanels=self.num_input_channels)
        opt = AdamW(network.parameters(), lr=self.lr)
        self.model = PoutyneModel(network, opt, self.loss, device=self.device)

    def train(self, train_dataset, val_dataset):
        model, train_needed = self.load_model(f"{self.name}-{self.latent_size}", self.model)
        self.model = model
        if train_needed:
            self.init_model()
            train_dataset_extracted = train_dataset[:]
            to_use_batch_size = min(len(val_dataset), self.batch_size)
            extracted_val = val_dataset[:]
            self.model.fit(train_dataset_extracted, train_dataset_extracted,
                           validation_data=(extracted_val, extracted_val),
                           epochs=500, verbose=self.verbose, batch_size=to_use_batch_size,
                           callbacks=[
                               poutyne.CosineAnnealingWarmRestarts(T_0=5, eta_min=1e-5),
                               EarlyStopping(patience=self.stop_patience, min_delta=1e-5),
                               BestModelRestore(verbose=False)],
                           dataloader_kwargs={"shuffle": True})
            self.save_model(f"{self.name}-{self.latent_size}", self.model)

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    def get_loss(self, x, reduction='mean'):
        prediction = self.predict(x)
        prediction = torch.tensor(prediction)
        reconstruction_error = self.loss(prediction, x, reduction=reduction)
        if x.shape[0] == 1:
            return reconstruction_error.item()
        return reconstruction_error
