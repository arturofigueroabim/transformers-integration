import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments,BertConfig 
from processing.datasets import create_dataset
from models.background_knowledge import CustomTrainer, create_libert_model
from processing.utils import compute_metrics, compute_class_weights
from evaluation.eval_report import run_classification_report
from config import CONFIG


# Load datasets
df = pd.read_pickle(CONFIG.data)
train_df = df[df['mode'] == 'train']
labels = train_df[train_df['label'].isin(['Attack', 'Support'])]['label']

class_weights, dic_class_weights = compute_class_weights(labels)

tokenizer =  BertTokenizerFast.from_pretrained(CONFIG.libert_pretrained_input_model)
train_dataset = create_dataset("train", tokenizer, False)
validate_dataset = create_dataset("validate", tokenizer, False)

libert_model, loss_fct = create_libert_model(num_labels=2, class_weights=class_weights)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=CONFIG.epochs,
    per_device_train_batch_size= CONFIG.batch_size,
    per_device_eval_batch_size=64,
    warmup_steps= CONFIG.warmup_steps,
    weight_decay=0.01,
    learning_rate=CONFIG.learning_rate,
    logging_steps=100000,
    #report_to='wandb',  # Enable logging to W&B
    #run_name='libert-fine-tuning',  # Name of the W&B run
)

trainer = CustomTrainer(
    model=libert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    compute_metrics=compute_metrics,
    loss_fct=loss_fct,
)

trainer.train()
run_classification_report(trainer, tokenizer)

# Save the model
trainer.save_model(CONFIG.libert_pretrained_output_model)