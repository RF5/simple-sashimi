dependencies = ['torch', 'torchaudio', 'einops', 'opt_einsum', 'fastprogress', 'omegaconf']

import torch
from pathlib import Path
from sashimi.model import Sashimi, SashimiAR
from omegaconf import OmegaConf


def sashimi_ar_sc09(pretrained=True, progress=True, device='cuda'):
    """ SaShiMi autoregressive model trained on SC09 dataset. """
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://github.com/RF5/simple-sashimi/releases/download/v1.0/ckpt_01100000.pt', 
        map_location=device, progress=progress
    )
    
    cfg = OmegaConf.create(checkpoint['cfg_yaml'])
    sashimi = SashimiAR(cfg.model_cfg).to(device)
    if pretrained:
        sashimi.load_state_dict(checkpoint['model_state_dict'])
    print(f"[MODEL] Sashimi loaded with {sum([p.numel() for p in sashimi.parameters()]):,d} parameters.")
    sashimi = sashimi.eval()
    return sashimi

    
    