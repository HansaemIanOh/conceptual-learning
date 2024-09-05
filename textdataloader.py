import os
import torch
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer

class TextDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            config,
            **kwargs,
        ) -> None:
        super().__init__()
        self.dataset_name = config.get('dataset_name')
        self.tokenizer_name = config.get('tokenizer_name')
        self.max_length = config.get('max_length', 128)
        self.batch_size = config.get('batch_size', 64)
        self.num_workers = config.get('num_workers', 4)
        self.train_val_test_split = config.get('train_val_test_split', (0.8, 0.1, 0.1))
        self.save_cache = config.get('save_cache')
        self.load_cache = config.get('load_cache')
        self.streaming = config.get('streaming')
        self.cac_pre = config.get('cac_pre')

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        if self.load_cache and os.path.exists(self.load_cache):
            rank_zero_info(f"Loading cached dataset from {self.load_cache}")
            self.tokenized_dataset = datasets.load_from_disk(self.load_cache)
            self.streaming = False
        else:
            rank_zero_info("Processing dataset...")
            dataset = datasets.load_dataset(self.dataset_name, streaming=self.streaming)

            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
    
            if self.streaming:
                self.tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            else:
                self.tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
                if "label" in self.tokenized_dataset["train"].features:
                    self.tokenized_dataset = self.tokenized_dataset.rename_column("label", "labels")
                self.save_processed_data(self.tokenized_dataset)

        if not self.streaming:
            self.tokenized_dataset.set_format("torch")
            self.split_dataset()
        else:
            self.prepare_streaming_dataset()

    def split_dataset(self):
        train_testvalid = self.tokenized_dataset["train"].train_test_split(
            test_size=self.train_val_test_split[1] + self.train_val_test_split[2]
        )
        test_valid = train_testvalid['test'].train_test_split(
            test_size=self.train_val_test_split[2] / (self.train_val_test_split[1] + self.train_val_test_split[2])
        )

        self.train_dataset = train_testvalid['train']
        self.valid_dataset = test_valid['train']
        self.test_dataset = test_valid['test']

    def prepare_streaming_dataset(self):
        def to_torch_tensors(example):
            return {k: torch.tensor(v) for k, v in example.items()}

        self.train_dataset = self.tokenized_dataset['train'].map(to_torch_tensors)
        if 'validation' in self.tokenized_dataset:
            self.valid_dataset = self.tokenized_dataset['validation'].map(to_torch_tensors)
        else:
            self.valid_dataset = self.tokenized_dataset['train'].take(1000).map(to_torch_tensors)
        if 'test' in self.tokenized_dataset:
            self.test_dataset = self.tokenized_dataset['test'].map(to_torch_tensors)
        else:
            self.test_dataset = None

    @rank_zero_only
    def save_processed_data(self, tokenized_dataset):
        if self.save_cache:
            rank_zero_info(f"Saving processed dataset to {self.save_cache}")
            tokenized_dataset.save_to_disk(self.save_cache)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=not self.streaming, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)