import os
import shutil
from safetensors.torch import save_file
from pytorch_lightning.callbacks import Callback
import torch

class SafetensorsCheckpoint(Callback):
    def __init__(
        self,
        config_path: str,
        dirpath: str,
        filename: str = "{epoch}-{step}-{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = False,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.config_path = config_path
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""
        self.last_model_path = ""

        torch_inf = torch.tensor(float('inf'))
        self.kth_value = torch_inf if self.mode == "min" else -torch_inf

    def _format_checkpoint_name(self, epoch, step, metrics):
        filename = self.filename.format(epoch=epoch, step=step, **metrics)
        return os.path.join(self.dirpath, f"{filename}.safetensors")

    def _save_checkpoint(self, trainer, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state_dict = trainer.lightning_module.state_dict()
        save_file(state_dict, filepath)

    def _update_best_and_save(self, current, filepath, trainer):
        if self.save_top_k == 0:
            return

        if len(self.best_k_models) < self.save_top_k:
            self.best_k_models[filepath] = current
            self._save_checkpoint(trainer, filepath)
        elif (self.mode == "min" and current < self.kth_value) or (self.mode == "max" and current > self.kth_value):
            if len(self.best_k_models) == self.save_top_k:
                del_filepath = self.kth_best_model_path
                self.best_k_models.pop(del_filepath)
                try:
                    if os.path.exists(del_filepath):
                        os.remove(del_filepath)
                except:
                    pass
            
            self.best_k_models[filepath] = current
            self._save_checkpoint(trainer, filepath)

        if self.mode == "min":
            self.kth_best_model_path = max(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]
            self.best_model_path = min(self.best_k_models, key=self.best_k_models.get)
        else:
            self.kth_best_model_path = min(self.best_k_models, key=self.best_k_models.get)
            self.kth_value = self.best_k_models[self.kth_best_model_path]
            self.best_model_path = max(self.best_k_models, key=self.best_k_models.get)
        
        self.best_model_score = self.best_k_models[self.best_model_path]

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics
        metric_value = metrics.get(self.monitor)

        if metric_value is None:
            return

        filepath = self._format_checkpoint_name(trainer.current_epoch, trainer.global_step, metrics)
        self._update_best_and_save(metric_value, filepath, trainer)

        if self.save_last:
            last_filepath = os.path.join(self.dirpath, "last.safetensors")
            self._save_checkpoint(trainer, last_filepath)
            self.last_model_path = last_filepath

    def on_train_start(self, trainer, pl_module):
            os.makedirs(self.dirpath, exist_ok=True)
            shutil.copy(self.config_path, os.path.join(self.dirpath, '../config.yaml'))