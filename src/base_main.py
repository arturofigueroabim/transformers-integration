import os
from config import CONFIG
from training.module import ClassificationModule
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from models.base import BaseClassificationModel
from evaluation.eval_report import run_classification_report

logger = WandbLogger(save_dir=os.getcwd(), name="base-model-fine-tuning")

model = BaseClassificationModel()

module = ClassificationModule(model)

trainer = pl.Trainer(accelerator = "gpu", 
                    devices = CONFIG.device_number, 
                    max_epochs=CONFIG.epochs,                    
                    logger=logger, 
                    fast_dev_run=CONFIG.fast_dev_run) 


trainer.fit(module)
#trainer.test(module)

# Save the model
trainer.save_checkpoint(CONFIG.pretrained_output_model)