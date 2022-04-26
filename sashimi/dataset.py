from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import random

class ARDataset(Dataset):

    def __init__(self, paths):
        super().__init__()
        self.filenames = paths
        logging.info(f"Dataset: found {len(self.filenames):,d} utterance files.")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx: int) -> Tensor:
        audio_filename = self.filenames[idx]
        signal, _ = torchaudio.load(audio_filename)
        out = signal.squeeze(0).clamp(-1.0, 1.0)
        
        return out

class ARCollate():
    def __init__(self, mu_quant_bins=256, seq_len=16000) -> None:
        self.seq_len = seq_len
        self.mu_tfm = torchaudio.transforms.MuLawEncoding(mu_quant_bins)

    def __call__(self, xs: List[Tensor]) -> Tuple[Tensor]:
        wavs = torch.zeros(len(xs), self.seq_len+1, dtype=torch.float)
        
        lengths = []
        for i in range(len(xs)):
            l = xs[i].shape[0]
            if l < self.seq_len + 1: # need to add one for next-token prediction
                lengths.append(l)
                wavs[i, :l] = xs[i][:l]
            else:
                start = random.randint(0, l - self.seq_len)
                wavs[i] = xs[i][start:start + self.seq_len]
                lengths.append(self.seq_len)
        
        signal = self.mu_tfm(wavs)
        lengths = torch.tensor(lengths).long()
        x = signal[:, :-1]
        y = signal[:, 1:]
        return x, y, lengths

        
