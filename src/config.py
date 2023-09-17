import os
from transformers import AutoTokenizer, BertTokenizerFast, RobertaTokenizer

class CONFIG:
    
    # BASE DIRECTORIES
    BASE_DIR = "./trained_model"
    BASE_MODEL_DIR = os.path.join(BASE_DIR, "base_models")
    ERNIE_MODEL_DIR = os.path.join(BASE_DIR, "ernie_models")
    KEPLER_MODEL_DIR = os.path.join(BASE_DIR, "kepler_models")
    LIBERT_MODEL_DIR = os.path.join(BASE_DIR, "libert_models")
    DATA_DIR = "./data"

    # PROJECT CONFIGURATIONS
    BASE_MODEL = {
        "base": {
            "name": "bert-base-uncased",
            "output_path": os.path.join(BASE_MODEL_DIR, "base_micro", "base_model.ckpt"),
            # "output_path_kialo": os.path.join(BASE_MODEL_DIR, "base_kialo", "base_model_kialo.ckpt")
        }
    }
    
    PRETRAINED_MODEL = {
        "ernie": {
            "name": "nghuyong/ernie-2.0-en",
            "tokenizer": BertTokenizerFast.from_pretrained("nghuyong/ernie-2.0-en"),
            "output_path": os.path.join(ERNIE_MODEL_DIR, "ernie_micro"),
            # "output_path": os.path.join(ERNIE_MODEL_DIR, "ernie_kialo")  # Change to ernie_kialo for kialo data
        },
        "kepler": {
            "input_path": os.path.join(KEPLER_MODEL_DIR, "kepler_input"),
            "tokenizer": RobertaTokenizer.from_pretrained(os.path.join(KEPLER_MODEL_DIR, "kepler_input")),
            "output_path": os.path.join(KEPLER_MODEL_DIR, "kepler_micro"), # Change to kepler_kialo for kialo data
            # "output_path": os.path.join(KEPLER_MODEL_DIR, "kepler_kialo") 
        },
        "libert": {
            "input_path": os.path.join(LIBERT_MODEL_DIR, "libert_input"),
            "tokenizer": BertTokenizerFast.from_pretrained(os.path.join(LIBERT_MODEL_DIR, "libert_input")),
            "output_path": os.path.join(LIBERT_MODEL_DIR, "libert_micro"),  # Change to libert_kialo for kialo data
            # "output_path": os.path.join(LIBERT_MODEL_DIR, "libert_kialo")
        }
    }
    
    COMBINATION_METHODS = ["concat", "average", "attention", "gated"]
    SAVE_METRICS_PATH = "results.csv"
    
    # MODEL PARAMETERS
    num_workers = 1
    batch_size = 32 #32 for kialo, 4 for microtext for a single GPU
    max_length = 128
    learning_rate = 3e-5
    warmup_steps = 100
    batch_accumulation = 1
    epochs = 5 #5 number of epochs

    DATA_PATH = os.path.join(DATA_DIR, "microtext_references.pickle")  # Change to kialo_references.pickle for kialo data
    DEVICE = "cuda"
    DEVICE_NUMBER = 1
    FAST_DEV_RUN = False  # If True, runs only 1 training and 1 validation example

    # SECRETS 
    OPENAI_SECRET = os.environ.get("OPENAI_SECRET", "DEFAULT_SECRET_HERE")
    WANDB_SECRET = os.environ.get("WANDB_SECRET", "DEFAULT_SECRET_HERE")