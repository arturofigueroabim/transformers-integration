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
        self.model = AutoModel.from_pretrained(CONFIG.BASE_MODEL["base"]["name"])
        
        self.hidden_size = self.model.config.hidden_size #768
        
        # (standard) model classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, n_classes),
            nn.Softmax(dim=1)
        )
        
        # initialize weights in linear layers
        self.init_weights(self.head)
        
        
    def init_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean = 0.0, std = 0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()
                    
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        
        output = self.model(input_ids = input_ids, 
                       token_type_ids = token_type_ids,
                       attention_mask = attention_mask,
                       output_hidden_states = True)
        
        # last hidden state of all tokens
        last_hidden_state = output.last_hidden_state
        
        return last_hidden_state[:, 0, :]  # Returns the hidden state of the first token in the sequence

    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        
        cls_hidden_state = self.forward(input_ids, attention_mask, token_type_ids)
        
        return self.head(cls_hidden_state)  # Returns the predicted classes