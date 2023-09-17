import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from models.combined import CombinedModel
from transformers import AutoTokenizer, BertTokenizerFast, RobertaTokenizer
from processing.datasets import RelationDataset
from torch.utils.data import DataLoader
from config import CONFIG
import pandas as pd

device = torch.device(CONFIG.DEVICE if torch.cuda.is_available() else "cpu")

def load_data():
    mapping = {'Attack': 0, 'Support': 1}
    df = pd.read_pickle(CONFIG.DATA_PATH_PATH)
    split = df[(df['mode'] == 'test') & (df['label'].isin(['Attack', 'Support']))]
    return split, split['label'].map(mapping)

def create_dataloader(dataset, tokenizer, batch_size):
    processed_dataset = RelationDataset(dataset[['premise', 'claim']], tokenizer)
    return DataLoader(processed_dataset, batch_size=batch_size)

def evaluate_model(model, base_dataloader, knowledge_dataloader):
    model.eval()
    predictions = []

    for (base_batch, knowledge_batch) in zip(base_dataloader, knowledge_dataloader):
        with torch.no_grad():
            base_batch = {k: v.to(device) for k, v in base_batch.items()}
            knowledge_batch = {k: v.to(device) for k, v in knowledge_batch.items()}

            outputs = model(base_batch, knowledge_batch)
        
        predictions.extend(torch.argmax(outputs, dim=1).tolist())

    return predictions

def get_classification_metrics(true_labels, predictions):
    report_dict = classification_report(true_labels, predictions, output_dict=True)

    report_dict["accuracy"] = {" ": accuracy_score(true_labels, predictions)}
    report_dict["precision"] = {" ": precision_score(true_labels, predictions)}
    report_dict["recall"] = {" ": recall_score(true_labels, predictions)}
    report_dict["f1_score"] = {" ": f1_score(true_labels, predictions)}

    return report_dict

def save_metrics(model_name, method, report_dict):
    df = pd.DataFrame(report_dict).transpose()
    df.index = df.index.astype(str) + f"_{method}"

    file_exists = os.path.isfile(model_name+CONFIG.SAVE_METRICS_PATH)
    df.to_csv(model_name + '_' + CONFIG.SAVE_METRICS_PATH, mode='a', header=not file_exists, index=True)

def main():
    # Load Data
    data_split, true_labels = load_data()

    # Tokenizers & Model Paths
    base_model_tokenizer = AutoTokenizer.from_pretrained(CONFIG.BASE_MODEL["base"]["name"])
    base_model_checkpoint = CONFIG.BASE_MODEL["base"]["output_path"]

    for model_name, model_info in CONFIG.PRETRAINED_MODEL.items():
        print(f"Evaluating {model_name} model...")

        knowledge_model_tokenizer = model_info["tokenizer"]
        knowledge_model_path = model_info["output_path"]
        
        for method in CONFIG.COMBINATION_METHODS:
            combined_model = CombinedModel(base_model_checkpoint, knowledge_model_path, method).to(device)
            
            for param in combined_model.base_model.parameters():
                param.requires_grad = False

            for param in combined_model.knowledge_model.parameters():
                param.requires_grad = False

            base_dataloader = create_dataloader(data_split, base_model_tokenizer, CONFIG.batch_size)
            knowledge_dataloader = create_dataloader(data_split, knowledge_model_tokenizer, CONFIG.batch_size)

            predictions = evaluate_model(combined_model, base_dataloader, knowledge_dataloader)

            report_dict = get_classification_metrics(true_labels, predictions)
            print(f"Method: {method}\nAccuracy: {report_dict['accuracy']}\nPrecision: {report_dict['precision']}\nRecall: {report_dict['recall']}\nF1-score: {report_dict['f1_score']}")

            save_metrics(model_name, method, report_dict)

if __name__ == "__main__":
    main()