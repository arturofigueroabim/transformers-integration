# RENAME FILE TO >>config.py<<
class CONFIG:
    # PROJECT
    experiment = "TEST BASE"
    # MODEL PARAMETER
    pretrained_model_name = "bert-base-uncased"
    num_workers = 1
    batch_size = 4 #32 for kialo, 4 for microtext for a single GPU
    max_length = 128
    learning_rate = 3e-5
    warmup_steps = 100
    batch_accumulation = 1
    epochs = 5 #5 number of epochs
    data = "../data/microtext_references.pickle"
    device = "cuda"
    device_number = 1
    fast_dev_run = True # if true: runs only 1 training and 1 validation example
    

    # SECRETS  
    openai_secret = "ENTER HERE"
    wandb_secret = "ENTER HERE"