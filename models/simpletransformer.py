import os
import numpy as np
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from safetensors.torch import save_file, load_file

class TBlock(nn.Module):
    def __init__(
        self,
        h_dim,
        num_heads,
        dropout=0.,
        attn_dropout=0,
        **kwargs
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        # LN -> MHA -> RES -> LN -> MLP -> RES
        self.LN1 = nn.LayerNorm(h_dim)
        self.MHA = nn.MultiheadAttention(h_dim, num_heads, dropout=attn_dropout, bias=False)
        self.LN2 = nn.LayerNorm(h_dim)
        self.MLP = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        '''
        x : [B, S, F]
        '''
        h = x
        h = self.LN1(h) # LN
        
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # MHA expects input of shape [S, B, F]
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        h, _ = self.MHA(q, k, v, attn_mask=attn_mask) # MHA
        h = h.transpose(0, 1) # Change back to [B, S, F]
        h = h + x # RES
        h = self.LN2(h) # LN
        h = self.MLP(h) # MLP
        h = h + x # RES

        return h

class SimpleTransformer(pl.LightningModule):
    def __init__(
        self, 
        config, 
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        '''
        Transformer block : Tensor -> Tensor
        h_dim : Hidden dimension of the transformer block
        num_heads : Number of heads in the transformer block
        dropout : Dropout rate for the MLP in the transformer block
        attn_dropout : Attention dropout rate in the transformer block
        '''
        self.h_dim = config['h_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.num_layers = config['num_layers']
        
        self.vocab_size = config['vocab_size']
        self.vocab_dim = config['vocab_dim']
        self.max_length = config['max_length']
        self.pad_token_id = config['pad_token_id']
        
        self.token_embed = nn.Embedding(self.vocab_size, self.vocab_dim)
        self.positional_embed = nn.Embedding(self.max_length, self.vocab_dim)
        # Transformer blocks
        
        self.TBlocks = nn.ModuleList()
        for index in range(self.num_layers):
            self.TBlocks.append(
                TBlock(
                    h_dim = self.h_dim, 
                    num_heads = self.num_heads,
                    dropout = self.dropout,
                    attn_dropout = self.attn_dropout
                )
            )
        
        # self.final = nn.Linear(self.h_dim, self.vocab_size)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)

        self.train_losses = []
        self.val_losses = []
    def forward(self, x: Tensor, attn_mask: Tensor=None) -> Tensor:
        '''
        x : [B, S]
        h : [B, S, F]
        output : [B, S, F] next token precision
        '''
        h = x
        position_ids = self.create_positional_indices(x)
        h = self.token_embed(h) + self.positional_embed(position_ids)
        
        if attn_mask==None:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=x.device)
        for module in self.TBlocks:
            h = module(h, attn_mask)
        # h = self.final(h)
        h = torch.einsum('BSF,FV->BSV', h, self.token_embed.weight.t())
        return h

    def create_positional_indices(self, x):
        batch_size, seq_length = x.size()
        
        # 시퀀스 길이만큼의 인덱스 생성
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        
        # 배치 차원으로 확장
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return position_ids

    def training_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        inputs = batch['input_ids']
        targets = inputs.clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.config['pad_token_id'] * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(inputs) # [B, S, vocab size]

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=self.config['pad_token_id'])
        preds = torch.argmax(logits, dim=2) # [B, S]
        mask = targets != self.config['pad_token_id']
        acc = (preds[mask] == targets[mask]).float().mean()

        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('TA', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        # 에포크 종료 시 평균 손실 계산 및 저장
        avg_loss = np.mean(self.train_losses)
        self.train_losses = []  # 리스트 초기화
        self.save_loss('train', avg_loss)

    def validation_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        inputs = batch['input_ids']
        targets = inputs.clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.config['pad_token_id'] * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(inputs) # [B, S, vocab size]
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=self.config['pad_token_id'])
        preds = torch.argmax(logits, dim=2) # [B, S, vocab size] -> [B, S]
        mask = targets != self.config['pad_token_id']
        acc = (preds[mask] == targets[mask]).float().mean()

        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('VA', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_losses)
        self.val_losses = []
        self.save_loss('val', avg_loss)

    def test_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        inputs = batch['input_ids']
        targets = inputs.clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.config['pad_token_id'] * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(inputs) # [B, S, vocab size]

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=self.config['pad_token_id'])
        preds = torch.argmax(logits, dim=2) # [B, S]
        mask = targets != self.config['pad_token_id']
        acc = (preds[mask] == targets[mask]).float().mean()

        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('TeA', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = \
        torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        try:
            if self.config['scheduler_gamma'] is not None:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = self.config['scheduler_gamma'])
                return {
                    "optimizer" : optimizer,
                    "lr_scheduler": {
                        "scheduler":scheduler,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }
                # return [optimizer], [scheduler]
        except:
            return optimizer

    def save_loss(self, phase, loss):
        save_path = os.path.join(self.logger.log_dir, f'{phase}_losses.npy')
        if self.trainer.is_global_zero:  # Preventing duplication for multi GPU environment
            try:
                losses = np.load(save_path)
                losses = np.append(losses, loss)
            except FileNotFoundError:
                losses = np.array([loss])
            np.save(save_path, losses)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config, *args, **kwargs):
        model = cls(config, *args, **kwargs)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return model