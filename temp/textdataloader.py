import os
from typing import *
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        if self.load_cache and os.path.exists(self.load_cache):
            rank_zero_info(f"Loading cached dataset from {self.load_cache}")
            tokenized_dataset = datasets.load_from_disk(self.load_cache)
        else:
            dataset = datasets.load_dataset(self.dataset_name)

            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
    
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])

            if "label" in tokenized_dataset["train"].features:
                tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

            self.save_processed_data(tokenized_dataset)

        self.total_size = sum(len(tokens) for split in tokenized_dataset.values() for tokens in split['input_ids'])

        tokenized_dataset.set_format("torch")

        # 데이터셋 분할
        train_testvalid = tokenized_dataset["train"].train_test_split(test_size=self.train_val_test_split[1] + self.train_val_test_split[2])
        test_valid = train_testvalid['test'].train_test_split(test_size=self.train_val_test_split[2] / (self.train_val_test_split[1] + self.train_val_test_split[2]))

        self.train_dataset = train_testvalid['train']
        self.valid_dataset = test_valid['train']
        self.test_dataset = test_valid['test']
        rank_zero_info("Dataset loaded")

    @rank_zero_only
    def save_processed_data(self, tokenized_dataset):
        rank_zero_info("Processing dataset...")
        if self.save_cache:
            rank_zero_info(f"Saving processed dataset to {self.save_cache}")
            tokenized_dataset.save_to_disk(self.save_cache)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

