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
        attn_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float=0.,
        attn_dropout: float=0,
        act: nn.Module=nn.GELU(),
        **kwargs
    ) -> None:
        super().__init__()
        self.act = act
        self.q_proj = nn.Linear(attn_dim, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.v_proj = nn.Linear(attn_dim, attn_dim)
        # LN -> MHA -> RES -> LN -> MLP -> RES
        self.LN1 = nn.LayerNorm(attn_dim)
        self.MHA = nn.MultiheadAttention(attn_dim, num_heads, dropout=attn_dropout, bias=False, batch_first=True)
        self.LN2 = nn.LayerNorm(attn_dim)
        self.MLP = nn.Sequential(
            nn.Linear(attn_dim, mlp_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, attn_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: Tensor, pad_attn_mask: Tensor, cas_attn_mask: Tensor) -> Tensor:
        '''
        x : [B, S, F]
        pad_attn_mask : [B, S]
        cas_attn_mask : [S, S]
        '''
        h = x
        h = self.LN1(h) # LN
        
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        h_attn, _ = self.MHA(q, k, v, key_padding_mask=pad_attn_mask, attn_mask=cas_attn_mask) # MHA
        h = h_attn + h # RES
        h = self.LN2(h) # LN
        h = self.MLP(h) # MLP
        h = h + x # RES

        return h

class CLCAC(pl.LightningModule):
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
        self.attn_dim = config['attn_dim']
        self.mlp_dim = config['mlp_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.num_layers = config['num_layers']
        self.tokenizer = config['tokenizer']
        self.vocab_dim = config['vocab_dim']
        self.max_length = config['max_length']
        
        self.token_embed = nn.Embedding(self.tokenizer.vocab_size, self.vocab_dim)
        self.positional_embed = nn.Embedding(self.max_length, self.vocab_dim)
        # Transformer blocks
        
        self.TBlocks = nn.ModuleList()
        for index in range(self.num_layers):
            self.TBlocks.append(
                TBlock(
                    attn_dim = self.attn_dim,
                    mlp_dim = self.mlp_dim, 
                    num_heads = self.num_heads,
                    dropout = self.dropout,
                    attn_dropout = self.attn_dropout
                )
            )
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)

        self.train_losses = []
        self.val_losses = []
    def forward(self, x: Tensor, pad_attn_mask: Tensor, cas_attn_mask: Tensor=None) -> Tensor:
        '''
        x : [B, S]
        pad_attn_mask : [B, S]
        cas_attn_mask : [S, S]
        h : [B, S, F]
        output : [B, S, F] next token precision
        '''
        h = x
        position_ids = self.create_positional_indices(x)
        h = self.token_embed(h) + self.positional_embed(position_ids)

        if cas_attn_mask==None:
            cas_attn_mask = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=x.device)
        pad_attn_mask = (1 - pad_attn_mask)
        pad_attn_mask = pad_attn_mask.float().masked_fill(pad_attn_mask == 1, float('-inf'))
        
        for module in self.TBlocks:
            h = module(h, pad_attn_mask, cas_attn_mask)

        h = torch.einsum('BSF,FV->BSV', h, self.token_embed.weight.t())
        return h

    def create_positional_indices(self, x):
        batch_size, seq_length = x.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return position_ids

    def random_masking(self, input_ids: Tensor, mask_prob: float = 0.15) -> Tensor:
        """
        Randomly mask input tokens for masked language modeling.
        
        Args:
        - input_ids (torch.Tensor): Input tensor of token ids. Shape: [batch_size, sequence_length]
        - mask_prob (float): Probability of masking a token. Default is 0.15 (15%)
        
        Returns:
        - torch.Tensor: Masked input tensor with the same shape as input_ids
        """
        masked_input_ids = input_ids.clone()
        
        # Create a mask for tokens that can be masked (exclude special tokens)
        can_mask = (input_ids != self.tokenizer.pad_token_id) & \
                (input_ids != self.tokenizer.cls_token_id) & \
                (input_ids != self.tokenizer.sep_token_id)
        
        # Create a random mask
        mask = (torch.rand(input_ids.shape, device=input_ids.device) < mask_prob) & can_mask
        
        # 80% of the time, replace with [MASK] token
        mask_80 = mask & (torch.rand(input_ids.shape, device=input_ids.device) < 0.8)
        masked_input_ids[mask_80] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random token
        mask_90 = mask & (torch.rand(input_ids.shape, device=input_ids.device) < 0.9) & ~mask_80
        random_tokens = torch.randint(self.tokenizer.vocab_size, input_ids.shape, device=input_ids.device)
        masked_input_ids[mask_90] = random_tokens[mask_90]
        
        # 10% of the time, keep the original token (done implicitly)
        return masked_input_ids, mask

    def training_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        input_ids = batch['input_ids']
        masked_ids, mask = self.random_masking(input_ids)
        logits = self(masked_ids, batch['attention_mask']) # [B, S, vocab size]

        pad_mask = input_ids != self.tokenizer.pad_token_id
        valid_mask = mask & pad_mask
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=-1) # [B, S, vocab size] -> [B, S]
        acc = (preds[valid_mask] == input_ids[valid_mask]).float().mean()
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('TA', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean(self.train_losses)
        self.train_losses = []
        self.save_loss('train', avg_loss)

    def validation_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        input_ids = batch['input_ids']
        masked_ids, mask = self.random_masking(input_ids)
        logits = self(masked_ids, batch['attention_mask']) # [B, S, vocab size]

        pad_mask = input_ids != self.tokenizer.pad_token_id
        valid_mask = mask & pad_mask
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=-1) # [B, S, vocab size] -> [B, S]
        acc = (preds[valid_mask] == input_ids[valid_mask]).float().mean()
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
        input_ids = batch['input_ids']
        masked_ids, mask = self.random_masking(input_ids)
        logits = self(masked_ids, batch['attention_mask']) # [B, S, vocab size]

        pad_mask = input_ids != self.tokenizer.pad_token_id
        valid_mask = mask & pad_mask
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), input_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=-1) # [B, S, vocab size] -> [B, S]
        acc = (preds[valid_mask] == input_ids[valid_mask]).float().mean()
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
                return [optimizer], [scheduler]
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