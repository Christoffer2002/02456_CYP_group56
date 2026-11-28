"""Pretrained MoLFormer encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from models.readouts import build_readout

class SMILESDataset(Dataset):
    def __init__(self, frame, targets):
        self.smiles = frame["canonical_smiles"].astype(str).tolist()
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return {"smiles": self.smiles[i], "y": torch.tensor(self.y[i])}


class MolFormerModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            cfg.checkpoint, deterministic_eval=True, trust_remote_code=True).to(device)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.config.hidden_size, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

        
    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def collate(self, batch):
        tokens = self.tokenizer([b["smiles"] for b in batch], padding=True, truncation=True, max_length=202, return_tensors="pt")
        return {**tokens, "y": torch.stack([b["y"] for b in batch])}

    def loader(self, frame, targets, batch_size, shuffle=False):
        return DataLoader(SMILESDataset(frame, targets), batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, num_workers=0)

    def param_groups(self):
        groups = [{"params": list(self.readout.parameters()), "lr": self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({"params": list(self.encoder.parameters()), "lr": self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        out = self.encoder(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device))
        return self.readout(out.pooler_output)