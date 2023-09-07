import torch
from config import CONFIG
from models.base import BaseClassificationModel
from training.module import ClassificationModule
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn

class CombinedModel(nn.Module):
    def __init__(self, base_model_checkpoint, knowledge_model_path, combination_method="concat", dropout=0.05, n_classes=2):
        super(CombinedModel, self).__init__()

        # Set the combionation model type
        self.combination_method = combination_method
        print(f"Combination Method Selected: " + combination_method)
                
        # Load the PyTorch trained base model
        # Initialize base model class
        self.base_model = BaseClassificationModel()
        
        # Initialize the ClassificationModule with the base model
        self.base_module = ClassificationModule(self.base_model)
        
        # Load the checkpoint
        checkpoint = torch.load(base_model_checkpoint)
        
        # Load the state dict into your base model
        self.base_model.load_state_dict(checkpoint['state_dict'], strict=False)
        base_output_size = self.base_model.hidden_size # Fetch output size from base model
        #print(f"base_output_size: {base_output_size}")

        # Load the pre-trained knowledge model
        knowledge_config = AutoConfig.from_pretrained(knowledge_model_path)
        self.knowledge_model = AutoModel.from_pretrained(knowledge_model_path, config=knowledge_config)
        knowledge_output_size = knowledge_config.hidden_size  # Fetch output size from knowledge model
        #print(f"knowledge_output_size: {knowledge_output_size}")

        # Initialize parameters for specific combination methods    
        if self.combination_method == "attention":
            self.attention_w = nn.Parameter(torch.randn(2))
        elif self.combination_method == "gated":
            self.gate_weight = nn.Parameter(torch.randn(1))
            
        # Calculate the combined output size
        if self.combination_method == "concat":
            combined_output_size = base_output_size + knowledge_output_size
        elif self.combination_method in ["average", "attention", "gated"]:
            combined_output_size = base_output_size      
                   
        print(f"combined_output_size: {combined_output_size}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_output_size, base_output_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(base_output_size, n_classes),
        )

    def forward(self, base_model_inputs, knowledge_model_inputs):
        # Pass input through base model
        base_output = self.base_model(**base_model_inputs)
        #print(f"base_output: {base_output.shape}")
        #print(f"base_output: {base_output.last_hidden_state}")

        # Pass input through knowledge model
        knowledge_output = self.knowledge_model(**knowledge_model_inputs)
        knowledge_hidden_state = knowledge_output.last_hidden_state[:, 0, :]
        #print(f"knowledge_hidden_state shape: {knowledge_hidden_state.shape}")
        #print(f"knowledge_hidden_state last_hidden_state: {len(knowledge_output.last_hidden_state)}")

        if self.combination_method == "concat":
            combined_output = torch.cat((base_output, knowledge_hidden_state), dim=1)
        elif self.combination_method == "average":
            combined_output = (base_output + knowledge_hidden_state) / 2
        elif self.combination_method == "attention":
            attention_weights = nn.Softmax(dim=0)(self.attention_w)
            combined_output = attention_weights[0] * base_output + attention_weights[1] * knowledge_hidden_state
        elif self.combination_method == "gated":
            gate = nn.Sigmoid()(self.gate_weight)
            combined_output = gate * base_output + (1 - gate) * knowledge_hidden_state
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    
        # Pass through final classification head
        return self.head(combined_output)

