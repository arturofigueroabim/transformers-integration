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
from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, AutoConfig

class ClassificationModule(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()

        self.model = model
        
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    def step(self, batch, batch_idx, mode):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        #x, y = batch
        #logits = self.forward(input_ids, attention_mask, token_type_ids )
        logits = self.model.predict(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        predictions = logits.argmax(dim = 1)
        
        loss = self.loss(logits, labels)
        accuracy = self.accuracy(predictions, labels)

        self.log(f'{mode}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        #logits = self(input_ids, attention_mask, token_type_ids)
        logits = self.model.predict(input_ids, attention_mask, token_type_ids)
        predictions = logits.argmax(dim=-1)
        
        return predictions

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
        split = split[split['label'] != 'Rephrase']
        
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.pretrained_model_name)
            
        return DataLoader(
            RelationDataset(split, tokenizer),
            batch_size = CONFIG.batch_size if mode == "train" else CONFIG.batch_size // 4,
            shuffle=shuffle, num_workers = CONFIG.num_workers
        )
    
    def train_dataloader(self):
        return self.create_data_loader(mode = "train", shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(mode = "validate")

    def test_dataloader(self):
        return self.create_data_loader(mode = "test")
