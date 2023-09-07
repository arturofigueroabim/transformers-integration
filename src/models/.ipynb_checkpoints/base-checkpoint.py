# TODO: generalize to [String] -> [Class] system 
# TODO: generalize forward pass
# TODO: set parameter that enables cls token utilization or arbitrary hidden layer utilization
# TODO: (MAYBE) generalize models to extend BASE, otherwise add 
from config import CONFIG

from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn

class BaseClassificationModel(nn.Module):
    
    def __init__(self, dropout = 0.05, n_classes = 2, injection = False):
        super(BaseClassificationModel, self).__init__()
        
        
        # model body
        self.model = AutoModel.from_pretrained(CONFIG.pretrained_model_name)
        
        self.hidden_size = self.model.config.hidden_size #768
        
        # (standard) model classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, n_classes)
        )
        
        # initialize weights in linear layers
        self.init_weights(self.head)
        
        
    def init_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean = 0.0, std = 0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()
                    
    
    def forward(self, x):
        
        input_ids, token_type_ids, attention_mask = x
        
        output = self.model(input_ids = input_ids, 
                       token_type_ids = token_type_ids,
                       attention_mask = attention_mask,
                       output_hidden_states = True)
        
        
        # last hidden state of all tokens
        last_hidden_state = output.last_hidden_state
        
        ################ Hidden States of each Transformer block
        ## Index=0 -> initial hidden state as token embedding + position embedding + segment embedding
        ## Index=13 -> last hidden state for each token in the sequence
        ## 
        ## "The ELMO authors suggest that lower levels encode syntax, while higher levels encode semantics."
        # hidden_states = output.hidden_states
        ################
        
        # hidden state of the first token e.g. classification token [CLS] or <s>
        # not a good representation of the whole sequence for decoder-based models such as GPT2
        cls_hidden_state = last_hidden_state[:, 0, :]
        
        # TODO: average hidden state of each token for each layer as better representation
        # TODO: consider earlier hidden states for syntax focused classification 
        return self.head(cls_hidden_state)