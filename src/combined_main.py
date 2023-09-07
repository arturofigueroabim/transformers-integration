import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.combined import CombinedModel
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizer
from transformers import BertTokenizerFast, TrainingArguments
from processing.datasets import RelationDataset
from torch.utils.data import Dataset, DataLoader
from config import CONFIG
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_combined_model(base_model_checkpoint, base_model_tokenizer, knowledge_model_path, knowledge_model_tokenizer, combination_method):
    combined_model = CombinedModel(base_model_checkpoint, knowledge_model_path, combination_method)

    # Freeze the parameters
    for param in combined_model.base_model.parameters():
        param.requires_grad = False

    for param in combined_model.knowledge_model.parameters():
        param.requires_grad = False

    combined_model = combined_model.to(device)

    # Load the test dataset
    mapping = {'Attack': 0, 'Support': 1}
    df = pd.read_pickle(CONFIG.data)
    split = df[(df['mode'] == 'test') & (df['label'].isin(['Attack', 'Support']))]
    true_labels = split['label'].map(mapping)

    # Prepare the datasets
    knowledge_dataset = RelationDataset(split[['premise', 'claim']], knowledge_model_tokenizer)
    base_dataset = RelationDataset(split[['premise', 'claim']], base_model_tokenizer)

    # Prepare the dataloaders
    batch_size = 16  # Choose an appropriate batch size for your environment
    base_dataloader = DataLoader(base_dataset, batch_size=batch_size)
    knowledge_dataloader = DataLoader(knowledge_dataset, batch_size=batch_size)

    combined_model.eval()  # Set the model to evaluation mode
    predictions = []

    # Iterate over batches from both dataloaders
    for (base_batch, knowledge_batch) in zip(base_dataloader, knowledge_dataloader):
        with torch.no_grad():
            # Move batch to device
            base_batch = {k: v.to(device) for k, v in base_batch.items()}
            knowledge_batch = {k: v.to(device) for k, v in knowledge_batch.items()}

            # Get model outputs
            outputs = combined_model(base_batch, knowledge_batch)
        
        # Get the predictions from the outputs
        predictions.extend(torch.argmax(outputs, dim=1).tolist())

    # Print classification report
    print(classification_report(true_labels, predictions))
    report_dict = classification_report(true_labels, predictions, output_dict=True)

    # Calculate and print other metrics
    report_dict["accuracy"] = {" ": accuracy_score(true_labels, predictions)}
    report_dict["precision"] = {" ": precision_score(true_labels, predictions)}
    report_dict["recall"] = {" ": recall_score(true_labels, predictions)}
    report_dict["f1_score"] = {" ": f1_score(true_labels, predictions)}
    
    print(f"Accuracy: {report_dict['accuracy']}")
    print(f"Precision: {report_dict['precision']}")
    print(f"Recall: {report_dict['recall']}")
    print(f"F1-score: {report_dict['f1_score']}")

    return report_dict

# Load Tokenizers
base_model_tokenizer = AutoTokenizer.from_pretrained(CONFIG.pretrained_model_name) #BASE Tokenizer
#knowledge_model_tokenizer = BertTokenizerFast.from_pretrained(CONFIG.ernie_pretrained_model_name) #ERNIE Tokenizer
# Alternative tokenizers can be uncommented based on the requirement
#knowledge_model_tokenizer = RobertaTokenizer.from_pretrained(CONFIG.kepler_pretrained_input_model) #KEPLER Tokenizer
knowledge_model_tokenizer = BertTokenizerFast.from_pretrained(CONFIG.libert_pretrained_input_model) #LIBERT Tokenizer

# Load Model Checkpoints
base_model_checkpoint = CONFIG.pretrained_output_model #BASE MODEL
#knowledge_model_path = CONFIG.ernie_pretrained_output_model #ERNIE MODEL
# Alternative model paths can be uncommented based on the requirement
#knowledge_model_path = CONFIG.kepler_pretrained_output_model #KEPLER MODEL
knowledge_model_path = CONFIG.libert_pretrained_output_model #LIBERT MODEL

# Evaluate models for each combination method
methods = CONFIG.combination_method
for method in methods:
    # Evaluate the combined model and store metrics
    report_dict = evaluate_combined_model(base_model_checkpoint, base_model_tokenizer, knowledge_model_path, knowledge_model_tokenizer, method)

    # Process the results and prepare for saving
    df = pd.DataFrame(report_dict).transpose()
    df.index = df.index.astype(str) + f"_{method}"

    # Check for existing metrics file and decide write mode
    file_exists = os.path.isfile(CONFIG.save_metrics)
    df.to_csv(CONFIG.save_metrics, mode='a', header=not file_exists, index=True)
