from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import random
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np

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

# -------------------------------
# Diffusion based code, 
# adapted from https://github.com/lmnt-com/diffwave

class UnconditionalDataset(Dataset):
    def __init__(self, paths):
        super().__init__()
        self.filenames = []
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)
        print(f"Dataset initialied with {len(self.filenames):,d} utterances.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = f'{audio_filename}.spec.npy'
        signal, _ = torchaudio.load(audio_filename)
        out = signal.squeeze(0).clamp(-1.0, 1.0)

        return {
                'audio': out,
                'spectrogram': None
        }

class DiffusionCollator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        for record in minibatch:
            if self.params.unconditional:
                    # Filter out records that aren't long enough.
                    if len(record['audio']) < self.params.audio_len:
                        if self.params.unconditional:
                            n_pad = self.params.audio_len - len(record['audio'])
                            record['audio'] = F.pad(record['audio'], (0, n_pad), mode='constant', value=0)
                        else:
                            del record['spectrogram']
                            del record['audio']
                        continue

                    start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
                    end = start + self.params.audio_len
                    record['audio'] = record['audio'][start:end]
                    record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
            else: raise NotImplementedError("Only unconditional sashimi available.")

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        if self.params.unconditional:
                return {
                        'audio': torch.from_numpy(audio),
                        'spectrogram': None,
                }
        spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        return {
                'audio': torch.from_numpy(audio),
                'spectrogram': torch.from_numpy(spectrogram),
        }

def from_path(data_dirs, params, is_distributed=False):
        dataset = UnconditionalDataset(data_dirs)

        return torch.utils.data.DataLoader(
                                dataset,
                                batch_size=params.batch_size,
                                collate_fn=DiffusionCollator(params).collate,
                                shuffle=not is_distributed,
                                num_workers=os.cpu_count(),
                                sampler=DistributedSampler(dataset) if is_distributed else None,
                                pin_memory=True,
                                drop_last=True)