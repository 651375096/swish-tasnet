import torch
from torch import nn


from asteroid.masknn import TDConvNet
masker =TDConvNet()


class MixtureConsistency(nn.Module):
    def __init__(self):
        super().__init__()


class GriffinLim(nn.Module):
    def __init__(self):
        super().__init__()


class MISI(nn.Module):
    def __init__(self):
        super().__init__()


from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from asteroid.data import LibriMix
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet

train_set, val_set = LibriMix
train_loader = DataLoader(train_set, batch_size=4, drop_last=True)
val_loader = DataLoader(val_set, batch_size=4, drop_last=True)

# Define model and optimizer
model = ConvTasNet(n_src=2)
optimizer = Adam(model.parameters(), lr=1e-3)
# Define Loss function.
loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
# Define System
system = System(model=model, loss_func=loss_func, optimizer=optimizer,
                train_loader=train_loader, val_loader=val_loader)
# Define lightning trainer, and train
trainer = pl.Trainer()
trainer.fit(system)
