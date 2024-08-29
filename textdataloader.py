import os
from typing import *
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            dataset_name: str,
            tokenizer_name: str,
            max_length: int = 128,
            batch_size: int = 32, 
            num_workers: int = 4,
            train_val_test_split: tuple = (0.8, 0.1, 0.1),
            **kwargs,
        ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.hansaem="It's me!!"
    def setup(self, stage: Optional[str] = None):
        # 전체 데이터셋 로드
        dataset = load_dataset(self.dataset_name)

        # 토큰화 함수 정의
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)
        
        # 데이터셋 토큰화
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        if "label" in tokenized_dataset["train"].features:
            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        
        self.total_size = sum(len(tokens) for split in tokenized_dataset.values() for tokens in split['input_ids'])

        tokenized_dataset.set_format("torch")

        # 데이터셋 분할
        train_testvalid = tokenized_dataset["train"].train_test_split(test_size=self.train_val_test_split[1] + self.train_val_test_split[2])
        test_valid = train_testvalid['test'].train_test_split(test_size=self.train_val_test_split[2] / (self.train_val_test_split[1] + self.train_val_test_split[2]))

        self.train_dataset = train_testvalid['train']
        self.valid_dataset = test_valid['train']
        self.test_dataset = test_valid['test']
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

