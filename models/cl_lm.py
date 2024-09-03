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
    cac(pre-training) -> gt(scr) -> cac(scr) -> cm(full-context)
    '''
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

        self.conceptual_dictionary = self.cd_func(self.concept_config) # [M, S_M]
        self.token_embed = nn.Embedding(self.tokenizer.vocab_size + 1, self.vocab_dim) # GT Token
        self.positional_embed = nn.Embedding(self.max_length, self.vocab_dim)

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
        
        h_concepts = self.cm_forward(
            h,
            self.cm,
            pad_attn_mask,
            cas_attn_mask
        )
        GT = torch.tensor(self.tokenizer.vocab_size).to(x.device)
        GT = self.token_embed(GT)
        scr = self.SCR(h, GT) # [B, M]
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
    ) -> Dict[str, nn.ModuleList]:
        models = {}
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
            models[model_name] = model_layers
        return models

    def cd_func(
        self, 
        concept_config: Dict
    ) -> Tensor:
        '''
        conceptual_dictionary : [M, S_CD] in cpu device
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
        return padded

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
        return h

    def cm_forward(
        self,
        h: Tensor,
        cm: Dict[str, nn.ModuleList],
        pad_attn_mask: Tensor, 
        cas_attn_mask: Tensor
    ) -> Dict[str, Tensor]:
        '''
        h : [B, S, F]
        return : [B, M, S, F]
        '''
        h_concepts = []
        for concept_name, modulelist in cm.items():
            h_concept = h
            for module in modulelist:
                h_concept = module(h_concept, pad_attn_mask, cas_attn_mask)
            h_concepts.append(h_concept)
        h_concepts = torch.stack(h_concepts, dim=1)
        return h_concepts

    def CR(
        self,
        query: Tensor,
        keys: Tensor
    ) -> Tensor:
        '''
        query : [B, F]
        keys : [B, M, F]
        return : [B, M]
        '''
        rho = F.softmax(torch.einsum('BF,BFM->BM', query, keys.transpose(1, 2)), dim=1)
        return rho

    def global_weight(
        self,
        query: Tensor,
        keys: Tensor
    ) -> Tensor:
        '''
        query : [B, F]
        keys : [B, S, F]
        return : [B, S]
        '''
        w = F.softmax(torch.einsum('BF,BFS->BS', query, keys.transpose(1, 2)), dim=1)
        return w

    def SCR(
        self,
        h: Tensor,
        GT: Tensor,
        pad_attn_mask: Tensor, 
        cas_attn_mask: Tensor
    ) -> Tensor:
        '''
        h : [B, S, F]
        GT : [B, F]
        queries : [B, S, F]
        conceptual_dictionary : [M, S_M]
        keys_cd : [B, M, S_M, F]
        keys_seq : [B, S, F]
        rho_seq : [B, S, M]
        w : [B, S]
        return : [B, M]
        '''
        conceptual_dictionary = self.conceptual_dictionary.to(h.device)
        for module in self.cac:
            q_proj = module.q_proj
            k_proj = module.k_proj
            break
        
        queries_seq = q_proj(h)
        query_GT = self.GT_q_proj(GT)
        keys_cd = k_proj(conceptual_dictionary)
        keys_seq = k_proj(h)

        rho_seq = F.softmax(torch.einsum('BSF,BFM->BSM', queries_seq, keys_cd.transpose(1, 2)), dim=2)
        w = self.global_weight(query_GT, keys_seq)
        scr = torch.einsum('BSM,BS->BM', rho_seq, w)
        return scr

    def create_positional_indices(self, x):
        batch_size, seq_length = x.size()
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        return position_ids

    def training_step(self, batch, batch_idx):
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
        targets = batch['input_ids'].clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.tokenizer.pad_token_id * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(batch['input_ids'], batch['attention_mask']) # [B, S, vocab size]
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=2) # [B, S, vocab size] -> [B, S]
        mask = targets != self.tokenizer.pad_token_id
        acc = (preds[mask] == targets[mask]).float().mean()

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
        targets = batch['input_ids'].clone()
        targets = targets[:, 1:]
        targets = torch.cat([targets, self.tokenizer.pad_token_id * torch.ones_like(targets[:, :1])], dim=1)
        logits = self(batch['input_ids'], batch['attention_mask']) # [B, S, vocab size]

        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1), ignore_index=self.tokenizer.pad_token_id)
        preds = torch.argmax(logits, dim=2) # [B, S]
        mask = targets != self.tokenizer.pad_token_id
        acc = (preds[mask] == targets[mask]).float().mean()

        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('TeA', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> List[Optimizer]:
        '''
        cac_optimizer, GT_optimizer, cm_optimizer
        '''
        common_params = list(self.token_embed.parameters()) + list(self.positional_embed.parameters())
        
        # CAC optimizer
        cac_params = list(self.cac.parameters()) + common_params
        cac_optimizer = torch.optim.Adam(
            cac_params,
            lr=self.config['cac_learning_rate'],
            weight_decay=self.config['cac_weight_decay']
        )

        # GT optimizer
        gt_params = list(self.GT_q_proj.parameters()) + common_params
        gt_optimizer = torch.optim.Adam(
            gt_params,
            lr=self.config['gt_learning_rate'],
            weight_decay=self.config['gt_weight_decay']
        )

        # CM optimizer
        cm_params = [p for model in self.cm.values() for p in model.parameters()] + common_params
        cm_optimizer = torch.optim.Adam(
            cm_params,
            lr=self.config['cm_learning_rate'],
            weight_decay=self.config['cm_weight_decay']
        )

        return [cac_optimizer, gt_optimizer, cm_optimizer]

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
