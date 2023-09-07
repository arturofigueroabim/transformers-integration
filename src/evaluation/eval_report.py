import pandas as pd
from processing.datasets import RelationDataset
from sklearn.metrics import classification_report
from config import CONFIG


def run_classification_report(trainer, tokenizer):
    # Load the test dataset
    mapping = {'Attack': 0, 'Support': 1}
    df = pd.read_pickle(CONFIG.data)
    split = df[(df['mode'] == 'test') & (df['label'].isin(['Attack', 'Support']))]
    split['label'] = split['label'].map(mapping)

    test_dataset = RelationDataset(split, tokenizer)

    # Make predictions
    raw_pred, _, _ = trainer.predict(test_dataset)
    preds = raw_pred.argmax(axis=1)

    # Print classification report
    report = classification_report(split['label'].values, preds)
    print(report)