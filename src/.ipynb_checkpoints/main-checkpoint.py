import os
os.environ["WANDB_MODE"] = "dryrun"

from config import CONFIG
from training.module import ClassificationModule
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from models.base import BaseClassificationModel

logger = WandbLogger(save_dir=os.getcwd(), name=CONFIG.experiment)

model = BaseClassificationModel()

module = ClassificationModule(model)

trainer = pl.Trainer(accelerator = "gpu", 
                    devices = CONFIG.device_number, 
                    max_epochs=CONFIG.epochs) 

trainer.fit(module)

trainer.test(module)

# Save the model
trainer.save_checkpoint("./trained_model/base_model.ckpt")