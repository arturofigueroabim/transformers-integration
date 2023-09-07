from transformers import BertForSequenceClassification, RobertaForSequenceClassification, RobertaConfig, BertConfig, Trainer
from torch.nn import CrossEntropyLoss
from config import CONFIG

class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, loss_fct, **kwargs):
        super().__init__(model, args, train_dataset=train_dataset, **kwargs)
        self.loss_fct = loss_fct

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def create_ernie_model(num_labels, class_weights):
    model = BertForSequenceClassification.from_pretrained(CONFIG.ernie_pretrained_model_name, 
                                                          num_labels=num_labels, 
                                                          output_attentions = False, 
                                                          output_hidden_states = False)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    return model, loss_fct

def create_kepler_model(num_labels, class_weights):
    config = RobertaConfig.from_pretrained(CONFIG.kepler_pretrained_input_model)
    model = RobertaForSequenceClassification.from_pretrained(CONFIG.kepler_pretrained_input_model, 
                                                            config= config)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    return model, loss_fct

def create_libert_model(num_labels, class_weights):
    config = BertConfig.from_pretrained(CONFIG.libert_pretrained_input_model)
    model = BertForSequenceClassification.from_pretrained(CONFIG.libert_pretrained_input_model,
                                                          config= config)
    loss_fct = CrossEntropyLoss(weight=class_weights)
    return model, loss_fct