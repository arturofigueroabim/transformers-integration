from config import CONFIG

# Program
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from transformers import AutoTokenizer
# TODO: generalize to [String] -> [Class] system 
# Pandas

class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, index):
        premise = self.data["premise"].iloc[index]
        claim = self.data["claim"].iloc[index]

        encoding = self.tokenizer.encode_plus(
            premise,
            claim,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )


        if 'label' in self.data.columns:
            
            label = torch.tensor(0 if self.data["label"].iloc[index] == "Attack" else 1, dtype=torch.int64)
            
            return {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
         }
            
        else:
            return {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
               

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    # TODO
    """TODO: implement alternative to default collate_fn to adjust max_length to longest sequence in batch"""
    pass

def create_dataset(mode: str, tokenizer, shuffle=False):
    
    df = pd.read_pickle(CONFIG.data)
    split = df[df['mode'] == mode]
    split = split[split['label'].isin(['Attack', 'Support'])]
    
    return RelationDataset(split, tokenizer)