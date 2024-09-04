import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from textdataloader import TextDataModule
from models import get_model
from torchinfo import summary
from safetensors.torch import save_file, load_file
from utils import SafetensorsCheckpoint
import shutil

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    dest='filename',
                    default='configs/flowers_cnn.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

data_module = TextDataModule(config['data_config'])

data_module.setup()

tokenizer = data_module.tokenizer

config['model_config']['tokenizer'] = tokenizer
config['model_config']['max_length'] = config['data_config']['max_length']
config['model_config']['streaming'] = config['data_config'].get('streaming')

# TensorBoard logger
logger = TensorBoardLogger(save_dir=config['log_config']['save_dir'], name=config['log_config']['name'])

seed_everything(config['model_config']['manual_seed'], True)

model = get_model(config['model_config']['name'], config['model_config'])

if config['log_config']['checkpoint']:
    try:
        current_version = int(logger.log_dir.split('version_')[-1])
        prev_version_dir = os.path.join(os.path.dirname(logger.log_dir), f'version_{current_version-1}')
        last_checkpoint = os.path.join(prev_version_dir, "checkpoints", "last.safetensors")
        if os.path.exists(last_checkpoint):
            model = model.__class__.load_from_checkpoint(last_checkpoint, config=config['model_config'])
            print(f"Loaded checkpoint from {last_checkpoint}")
        else:
            print(f"No checkpoint found at {last_checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


# Checkpoint callback
safetensors_callback = SafetensorsCheckpoint(
    config_path=args.filename,
    dirpath=os.path.join(logger.log_dir, "checkpoints"),
    monitor='VL',
    mode='min',
    save_top_k=2,
    save_last=True,
    filename='epoch={epoch}-VL={VL:.4f}'
)

trainer = pl.Trainer(
    enable_checkpointing=False,
    logger=logger,
    callbacks=[LearningRateMonitor(),
               safetensors_callback],
    log_every_n_steps=1,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    **config['trainer_config']
)

trainer.fit(model, datamodule=data_module)

@rank_zero_only
def test_model():
    test_trainer = pl.Trainer(devices=1, num_nodes=1)
    test_trainer.test(model, data_module.test_dataloader(), verbose=True)

@rank_zero_only
def cleanup_logs():
    if os.path.exists('lightning_logs'):
        shutil.rmtree('lightning_logs')

test_model()
cleanup_logs()