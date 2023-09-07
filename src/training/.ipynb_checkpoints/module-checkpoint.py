from config import CONFIG
from processing.datasets import RelationDataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List
import pandas

import pytorch_lightning as pl
import torchmetrics 
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup


class ClassificationModule(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()

        self.model = model
        
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x, *args):
        return self.model(x, *args)
        
    def step(self, batch, batch_idx, mode):
        x, y = batch
        logits = self.forward(x)

        predictions = logits.argmax(dim = 1)
        
        loss = self.loss(logits, y)
        accuracy = self.accuracy(predictions, y)

        self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=CONFIG.learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CONFIG.warmup_steps,
            num_training_steps=len(self.train_dataloader().dataset) // CONFIG.batch_size * CONFIG.epochs,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def create_data_loader(self, mode: str, shuffle=False):
        df = pandas.read_pickle(CONFIG.data)
        split = df[df['mode'] == mode]
        
        return DataLoader(
            RelationDataset(split),
            batch_size = CONFIG.batch_size if mode == "train" else CONFIG.batch_size // 4,
            shuffle=shuffle, num_workers = CONFIG.num_workers
        )
    
    def train_dataloader(self):
        return self.create_data_loader(mode = "train", shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(mode = "validate")

    def test_dataloader(self):
        return self.create_data_loader(mode = "test")
