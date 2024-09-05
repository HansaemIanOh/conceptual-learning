import os
import numpy as np
from typing import *
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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

class ConceptualLM(pl.LightningModule):
    '''
    cac_pre -> gt_scr -> cac_scr -> cm
    '''
    def __init__(
        self, 
        config, 
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        '''
        training_mode : cac_pre, gt_scr, cac_scr, cm
        Transformer block : Tensor -> Tensor
        h_dim : Hidden dimension of the transformer block
        num_heads : Number of heads in the transformer block
        dropout : Dropout rate for the MLP in the transformer block
        attn_dropout : Attention dropout rate in the transformer block
        '''
        self.attn_dim = config.get('attn_dim', 512)
        self.mlp_dim = config.get('mlp_dim', 2048)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.0)
        self.attn_dropout = config.get('attn_dropout', 0.0)
        self.num_layers = config.get('num_layers', 6)
        self.tokenizer = config.get('tokenizer', None)
        self.vocab_dim = config.get('vocab_dim', None)
        self.max_length = config.get('max_length', 512)
        self.streaming = config.get('streaming')
        self.concept_config = config.get('concept_config')
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        self.training_dict = config.get('training_mode')
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        self.conceptual_dictionary, self.cd_padding_mask = self.cd_func(self.concept_config) # [M, T]
        self.token_embed = nn.Embedding(self.tokenizer.vocab_size, self.vocab_dim)
        self.positional_embed = nn.Embedding(self.max_length, self.vocab_dim)
        self.GT_embed = nn.Embedding(1, self.vocab_dim) # GT Token
        self.cac = self.cac_func(config.get('cac_config'))
        self.cm = self.cm_func(self.concept_config)
        GT_attn_dim = config.get('cac_config').get('attn_dim')
        self.GT_q_proj = nn.Linear(GT_attn_dim, GT_attn_dim)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size)

        self.train_losses = []
        self.val_losses = []
        if self.streaming is not None:
            self.step_count = 0
            self.val_step_count = 0

    def forward(
        self, 
        x: Tensor, 
        pad_attn_mask: Tensor, 
        cas_attn_mask: Tensor=None
    ) -> Tensor:
        '''
        x : [B, S] -> UW_e + W_p
        CD : [c_1, ..., c_M]
        c_j -tokenizing> [0, ..., 2] 0 : SOS, 2 : EOS
        pad_attn_mask : [B, S] -> [B, 1, S]
        cas_attn_mask : [S, S] -> [1, S, S]
        mask : [B, S, S]
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
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        if self.training_dict=='cac_pre':
            return self.cac_forward(
                h,
                self.cac,
                pad_attn_mask,
                cas_attn_mask
            )
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        GT = torch.tensor(0).to(x.device)
        GT = self.GT_embed(GT)
        scr = self.Multi_SCR(h, GT, pad_attn_mask) # [B, M]
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        if self.training_dict in ['gt_scr', 'cac_scr']:
            return scr
        # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        h_concepts = self.cm_forward(
            h,
            self.cm,
            pad_attn_mask,
            cas_attn_mask
        )
        h = torch.einsum('BMSF,BM->BSF', h_concepts, scr)
        h = torch.einsum('BSF,FV->BSV', h, self.token_embed.weight.t())
        return h

    def cac_func(
        self,
        config: Dict
    ) -> nn.ModuleList:
        models = nn.ModuleList()
        for index in range(config.get('num_layers')):
            models.append(
                TBlock(
                    attn_dim = config.get('attn_dim'),
                    mlp_dim = config.get('mlp_dim'),
                    num_heads = config.get('num_heads'),
                    dropout = config.get('dropout'),
                    attn_dropout = config.get('attn_dropout')
                )
            )
        return models

    def cm_func(
        self, 
        config: Dict
    ) -> nn.ModuleList:
        models = nn.ModuleList()
        for model_name, model_config in config.items():
            model_layers = nn.ModuleList()
            for _ in range(model_config.get('num_layers')):
                model_layers.append(
                    TBlock(
                        attn_dim=model_config.get('attn_dim'),
                        mlp_dim=model_config.get('mlp_dim'),
                        num_heads=model_config.get('num_heads'),
                        dropout=model_config.get('dropout'),
                        attn_dropout=model_config.get('attn_dropout')
                    )
                )
            models.append(model_layers)
        return models

    def cd_func(
        self, 
        concept_config: Dict
    ) -> Tensor:
        '''
        conceptual_dictionary : [M, T] in cpu device
        padding_mask : [M, T] in cpu device
        '''
        conceptual_dictionary = []
        for models in concept_config:
            token = concept_config.get(models).get('token')
            conceptual_dictionary.append(token)

        tokenized = [torch.tensor(self.tokenizer.encode(
            token, 
            add_special_tokens=False, 
            padding=False, 
            truncation=True))
            for token in conceptual_dictionary]
        padded = pad_sequence(tokenized, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        padding_mask = torch.zeros_like(padded, dtype=torch.float)
        padding_mask[padded == self.tokenizer.pad_token_id] = float('-inf')
        return padded, padding_mask

    def cac_forward(
        self,
        h: Tensor,
        modulelist: nn.ModuleList,
        pad_attn_mask: Tensor, 
        cas_attn_mask: Tensor
    ) -> Tensor:
        '''
        h : [B, S, F]
        return : [B, S, V]
        For pre-training cac
        '''
        for module in modulelist:
            h = module(h, pad_attn_mask, cas_attn_mask)
        h = torch.einsum('BSF,FV->BSV', h, self.token_embed.weight.t())
        return h

    def cm_forward(
        self,
        h: Tensor,
        cm: nn.ModuleList,
        pad_attn_mask: Tensor, 
        cas_attn_mask: Tensor
    ) -> Dict[str, Tensor]:
        '''
        h : [B, S, F]
        return : [B, M, S, F]
        '''
        h_concepts = []
        for modulelist in cm:
            h_concept = h
            for module in modulelist:
                h_concept = module(h_concept, pad_attn_mask, cas_attn_mask)
            h_concepts.append(h_concept)
        h_concepts = torch.stack(h_concepts, dim=1)
        return h_concepts

    def Multi_SCR(
        self,
        h: Tensor,
        GT: Tensor,
        pad_attn_mask: Tensor, 
    ) -> Tensor:
        '''
        h : [B, S, F] # pad_seq
        GT : [F]
        pad_attn_mask : [B, S]
        queries_seq : [B, S, F] # pad_seq
        conceptual_dictionary : [M, T] # pad_cd
        keys_cd : [M, T, F] # pad_cd
        keys_seq : [B, S, F] # pad_seq
        cd_padding_mask : [M, T]
        w_cd : [M, T] # pad_cd
        w_seq : [B, S] # pad_seq
        mcr : [B, S, M, T] # pad_seq
        rho_seq : [B, S, M] # pad_seq
        return : [B, M]
        '''
        conceptual_dictionary = self.conceptual_dictionary.to(h.device)
        cd_embed = self.token_embed(conceptual_dictionary)
        cd_padding_mask = self.cd_padding_mask.to(h.device)
        for module in self.cac:
            q_proj = module.q_proj
            k_proj = module.k_proj
            break
        
        queries_seq = q_proj(h)
        query_GT = self.GT_q_proj(GT)
        keys_cd = k_proj(cd_embed)
        keys_seq = k_proj(h)
        w_cd = F.softmax(self.apply_mask(
            torch.einsum(
                'F,MTF->MT',query_GT, keys_cd
                ), attn_mask = cd_padding_mask), dim=1)

        mcr = torch.einsum('BSF,MTF->BSMT', queries_seq, keys_cd)

        rho_seq = F.softmax(torch.einsum('BSMT,MT->BSM',mcr, w_cd), dim=2)

        w_seq = F.softmax(self.apply_mask(
            torch.einsum(
                'F,BFS->BS', query_GT, keys_seq.transpose(1, 2)
                ), attn_mask = pad_attn_mask), dim=1)
        scr = torch.einsum('BSM,BS->BM', rho_seq, w_seq)
        return scr

    def apply_mask(
        self,
        tensor: Tensor,
        attn_mask: Tensor
    ) -> Tensor:
        return tensor + attn_mask

    def create_positional_indices(self, x):
        batch_size, seq_length = x.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return position_ids

    def loss_cac_pre(self, batch):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        targets = batch['input_ids'].clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.tokenizer.pad_token_id * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(batch['input_ids'], batch['attention_mask']) # [B, S, V]

        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=2) # [B, S]
        mask = targets != self.tokenizer.pad_token_id
        acc = (preds[mask] == targets[mask]).float().mean()

        return loss, acc

    def loss_scr(self, batch):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask, targets
        input_ids : [B, S]
        logits : [B, M]
        targets : [B] (0, ..., M-1)
        '''
        targets = batch['labels']
        logits = self(batch['input_ids'], batch['attention_mask']) # [B, M]

        loss = F.cross_entropy(logits, targets, ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=1) # [B]
        mask = targets != self.tokenizer.pad_token_id
        acc = (preds[mask] == targets[mask]).float().mean()
        return loss, acc

    def loss_cm(self, batch):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        targets = batch['input_ids'].clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.tokenizer.pad_token_id * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(batch['input_ids'], batch['attention_mask']) # [B, S, vocab size]

        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=2) # [B, S]
        mask = targets != self.tokenizer.pad_token_id
        acc = (preds[mask] == targets[mask]).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        arxiv batch : labels, input_ids, attention_mask
        input_ids : [B, S]
        '''

        if self.training_dict=='cac_pre':
            loss, acc = self.loss_cac_pre(batch)
        if self.training_dict=='gt_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cac_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cm':
            loss, acc = self.loss_cm(batch)

        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('TA', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss.item())
        if self.streaming is not None:
            self.step_count += 1

            if self.step_count % 1000 == 0:
                avg_loss = np.mean(self.train_losses)
                self.save_loss('train', avg_loss)
                self.train_losses = []

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
        if self.training_dict=='cac_pre':
            loss, acc = self.loss_cac_pre(batch)
        if self.training_dict=='gt_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cac_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cm':
            loss, acc = self.loss_cm(batch)

        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('VA', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_losses.append(loss.item())
        if self.streaming is not None:
            self.val_step_count += 1

            if self.val_step_count % 100 == 0:
                avg_loss = np.mean(self.val_losses)
                self.save_loss('val', avg_loss)
                self.val_losses = []

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_losses)
        self.val_losses = []
        self.save_loss('val', avg_loss)

    def test_step(self, batch, batch_idx):
        '''
        batch : labels, input_ids, token_type_ids, attention_mask
        input_ids : [B, S]
        '''
        if self.training_dict=='cac_pre':
            loss, acc = self.loss_cac_pre(batch)
        if self.training_dict=='gt_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cac_scr':
            loss, acc = self.loss_scr(batch)
        if self.training_dict=='cm':
            loss, acc = self.loss_cm(batch)

        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('TeA', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> List[Optimizer]:
        '''
        cac_optimizer, gt_optimizer, cm_optimizer, all_optimizer
        '''
        common_params = list(self.token_embed.parameters()) + list(self.positional_embed.parameters())
        gt_params = list(self.GT_q_proj.parameters())

        # GT optimizer
        gt_src_params = gt_params + common_params
        gt_optimizer = torch.optim.Adam(
            gt_src_params,
            lr=self.config['gt_learning_rate'],
            weight_decay=self.config['gt_weight_decay']
        )

        # CAC optimizer
        cac_params = list(self.cac.parameters()) + gt_params + common_params
        cac_optimizer = torch.optim.Adam(
            cac_params,
            lr=self.config['cac_learning_rate'],
            weight_decay=self.config['cac_weight_decay']
        )

        # CM optimizer
        cm_params = [p for model in self.cm for p in model.parameters()] + gt_params + common_params
        cm_optimizer = torch.optim.Adam(
            cm_params,
            lr=self.config['cm_learning_rate'],
            weight_decay=self.config['cm_weight_decay']
        )

        # All optimizer
        # all_optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.config['cm_learning_rate'],
        #     weight_decay=self.config['cm_weight_decay']
        # )
        optimizer_dict = {
            'cac_pre': cac_optimizer,
            'gt_scr': gt_optimizer,
            'cac_scr': cac_optimizer,
            'cm': cm_optimizer
        }
        optimizer = optimizer_dict.get(self.training_dict)
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
