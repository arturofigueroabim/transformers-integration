# RENAME FILE TO >>config.py<<
class CONFIG:
    # PROJECT
    pretrained_model_name = "bert-base-uncased"
    pretrained_output_model = "./trained_model/base_models/base_micro/base_model.ckpt"
    #pretrained_output_model = "./trained_model/base_models/base_kialo/base_model_kialo.ckpt"

    ernie_pretrained_model_name = "nghuyong/ernie-2.0-en"
    #ernie_pretrained_output_model = "./trained_model/ernie_models/ernie_micro/"
    ernie_pretrained_output_model = "./trained_model/ernie_models/ernie_kialo/"

    kepler_pretrained_input_model = "./trained_model/kepler_models/kepler_input/"
    #kepler_pretrained_output_model =  "./trained_model/kepler_models/kepler_micro/"
    kepler_pretrained_output_model =  "./trained_model/kepler_models/kepler_kialo/"

    libert_pretrained_input_model = "./trained_model/libert_models/libert_input/"
    libert_pretrained_output_model = "./trained_model/libert_models/libert_micro/"
    #libert_pretrained_output_model = "./trained_model/libert_models/libert_kialo_2/"

    combination_method = ["concat", "average", "attention", "gated"] 
    
    save_metrics = "b_l_results.csv"
    
    # MODEL PARAMETER
    num_workers = 1
    batch_size = 32 #32 for kialo, 4 for microtext for a single GPU
    max_length = 128
    learning_rate = 3e-5
    warmup_steps = 100
    batch_accumulation = 1
    epochs = 5 #5 number of epochs
    
    #data = "../data/microtext_references.pickle" # Create on for kialo
    data = "./data/kialo_references.pickle" 
    
    device = "cuda"
    device_number = 1
    fast_dev_run = False # if true: runs only 1 training and 1 validation example
    

    # SECRETS  
    openai_secret = "ENTER HERE"
    wandb_secret = "78d69a339f5c9e47e83b23695b39e1f41fbe1fb3"