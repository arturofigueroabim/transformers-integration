from config import CONFIG

# Program
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
import pandas 

from transformers import AutoTokenizer
# TODO: generalize to [String] -> [Class] system 
# Pandas



class RelationDataset(Dataset):
    """
    Dataset Class for the Argument Relation Classification Task
    """
    def __init__(self, data):
        assert type(data) == pandas.DataFrame
        
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.pretrained_model_name)
        
    def __getitem__(self, index):

        premise = self.data["premise"].iloc[index]
        claim = self.data["claim"].iloc[index]

        ## Tokenize and encode the strings using the selected BERT model
        # both strings are encoded individually to receive encodings of the form [CLF] [Sx...] [SEP] twice 
        # instead of [CLF] [S1...] [SEP] [S2...] [SEP]
        # TODO: compare both approaches
        
        context_encoding = self.tokenizer.encode_plus(
            premise,             # the sentence to be encoded
            claim,
            add_special_tokens = True,  # Add [CLS] and [SEP]
            max_length = CONFIG.max_length,          # maximum length of a sentence
            padding='max_length',
            truncation=True,   # Add [PAD]s
            return_token_type_ids = True,
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',   # ask the function to return PyTorch tensors
        )

        context_encoding = (context_encoding['input_ids'].flatten(), 
                         context_encoding['token_type_ids'].flatten(),
                         context_encoding['attention_mask'].flatten())
        
        # 1. Keywords identifizierebn: Das auto ist schlecht für die umwelt -> auto, umwelt 
        # 2. Wordnet Text Embeddings: def(auto) -> text embedding 
        # 2. Knowledge Embeddings: Auto-benötigt->Treibstoff --> Embedding 
        
        #premise(keywords) - premise(keywords)
        
        #premise(keywords) - claim(keywords)
        
        # claim(keywords) - claim(keywords)
        
        # TODO: knowledge_encoding
        
        
        # Get the label
        label = torch.tensor(0 if self.data["label"].iloc[index] == "Attack" else 1, dtype=torch.int64)
        
        return (context_encoding,
                torch.tensor(label, dtype=torch.int64))

    
    def __len__(self):
        return len(self.data)

    
def collate_fn(batch):
    # TODO
    """TODO: implement alternative to default collate_fn to adjust max_length to longest sequence in batch"""
    pass