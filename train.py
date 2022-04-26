import argparse
import logging
import os
import math
import random
import sched
import time
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from omegaconf import MISSING, OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from sashimi.config import AutoregressiveConfig
from sashimi.dataset import ARCollate, ARDataset
from sashimi.model import SashimiAR


@dataclass
class DistributedConfig:
    dist_backend: str = 'nccl'
    dist_url: str = "tcp://localhost:54321"
    n_nodes: int = 1
    n_gpus_per_node: int = 1

@dataclass
class TrainConfig:
    # Distributed settings
    distributed: DistributedConfig = DistributedConfig()
    # Model settings
    model_cfg: AutoregressiveConfig = AutoregressiveConfig()

    device: str = 'cuda'
    seed: int = 1775
    
    batch_size: int = 8
    num_workers: int = 8
    fp16: bool = True
    max_steps: int = 1_100_000 # 1.1M steps for SC09 
    summary_interval: int = 25
    checkpoint_interval: int = 2500
    stdout_interval: int = 100
    validation_interval: int = 1000

    # Learning settings
    start_lr: float = 4e-3 # unspecified for sashimi in paper. Using their config. 
    # plateau lr schedule settings
    plateau_mode: str = "min"
    plateau_factor: float = 0.2 
    plateau_patience: int = 20 
    plateau_min_lr: float = 0.0
    grad_clip: float = 0 # disabled
    
    # Data settings
    checkpoint_path: str = MISSING
    train_csv: str = MISSING
    valid_csv: str = MISSING
    resume_checkpoint: str = ''
    sample_rate: int = 16000
    seq_len: int = 16000


def flatten_cfg(cfg: Union[DictConfig, ListConfig]) -> dict:
    """ 
    Recursively flattens a config into a flat dictionary compatible with 
    tensorboard's `add_hparams` function.
    """
    out_dict = {}
    if type(cfg) == ListConfig:
        cfg = DictConfig({f"[{i}]": v for i, v in enumerate(cfg)})

    for key in cfg:
        if type(getattr(cfg, key)) in (int, str, bool, float):
            out_dict[key] = getattr(cfg, key)
        elif type(getattr(cfg, key)) in [DictConfig, ListConfig]:
            out_dict = out_dict | {f"{key}{'.' if type(getattr(cfg, key)) == DictConfig else ''}{k}": v for k, v in flatten_cfg(getattr(cfg, key)).items()}
        else: raise AssertionError
    return out_dict

def train(rank, cfg: TrainConfig):
    if cfg.distributed.n_gpus_per_node > 1:
        init_process_group(backend=cfg.distributed.dist_backend, init_method=cfg.distributed.dist_url,
                           world_size=cfg.distributed.n_nodes*cfg.distributed.n_gpus_per_node, rank=rank)

    device = torch.device(f'cuda:{rank:d}')

    model = SashimiAR(cfg.model_cfg).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    
    logging.info(f"Initialized rank {rank}")
    
    if rank == 0:
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Model initialized as:\n {model}")
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory : {cfg.checkpoint_path}")
        logging.info(f"Model has {sum([p.numel() for p in model.parameters()]):,d} parameters.")

    steps = 0
    if cfg.resume_checkpoint != '' and os.path.isfile(cfg.resume_checkpoint):
        state_dict = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        steps = state_dict['steps'] + 1
        last_epoch = state_dict['epoch']
        print(f"Checkpoint loaded from {cfg.resume_checkpoint}. Resuming training from {steps} steps at epoch {last_epoch}")
    else:
        state_dict = None
        last_epoch = -1

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
        if rank == 0: logging.info("Multi-gpu detected")
        model = DDP(model, device_ids=[rank]).to(device)

    optim = torch.optim.AdamW(chain(model.parameters(), loss_fn.parameters()), cfg.start_lr, weight_decay=0)
    if state_dict is not None: optim.load_state_dict(state_dict['optim_state_dict'])

    train_df, valid_df = pd.read_csv(cfg.train_csv), pd.read_csv(cfg.valid_csv)

    trainset = ARDataset(train_df.path.tolist())

    train_sampler = DistributedSampler(trainset) if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else None

    train_loader = DataLoader(trainset, num_workers=cfg.num_workers, 
                              shuffle=False if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else True,
                              sampler=train_sampler,
                              batch_size=cfg.batch_size,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=ARCollate(cfg.model_cfg.mu_levels, cfg.seq_len))

    if rank == 0:
        validset = ARDataset(valid_df.path.tolist())
        validation_loader = DataLoader(validset, num_workers=cfg.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=cfg.batch_size,
                                       pin_memory=False,
                                       drop_last=True,
                                       collate_fn=ARCollate(cfg.model_cfg.mu_levels, cfg.seq_len))

        sw = SummaryWriter(os.path.join(cfg.checkpoint_path, 'logs'))


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode=cfg.plateau_mode, factor=cfg.plateau_factor,
                                                            patience=cfg.plateau_patience, min_lr=cfg.plateau_min_lr)

    if state_dict is not None:
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    if cfg.fp16: 
        scaler = GradScaler()
        if state_dict is not None and 'scaler_state_dict' in state_dict:
            scaler.load_state_dict(state_dict['scaler_state_dict'])

    model.train()
    
    if rank == 0: 
        max_epochs = math.ceil(cfg.max_steps/len(train_loader))
        mb = master_bar(range(max(0, last_epoch), max_epochs))
        sw.add_text('config', '```\n' + OmegaConf.to_yaml(cfg) + '\n```', global_step=steps)
        smooth_loss = None
    else: mb = range(max(0, last_epoch), max_epochs)    

    for epoch in mb:
        if rank == 0:
            start = time.time()
            mb.write("Epoch: {}".format(epoch+1))

        if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
            train_sampler.set_epoch(epoch)

        if rank == 0: pb = progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb)
        else: pb = enumerate(train_loader)

        if steps > cfg.max_steps: break
        
        for i, batch in pb:
            if rank == 0: start_b = time.time()
            x, y, lens = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True) # (bs, seq_len)
            lens = lens.to(device, non_blocking=True)
            
            optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                logits = model(x, lens) # (bs, seq_len, mu_levels)
                # print(y.shape, logits.shape)
                logits = logits.view(-1, cfg.model_cfg.mu_levels) # reshape for CE loss (N, C)
                y_ = y.view(-1)
                loss = loss_fn(logits, y_)
            if cfg.fp16: 
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                if cfg.grad_clip > 0:
                    gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                else: 
                    gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in model.parameters()]), 2)
                scaler.step(optim)
                scaler.update()
            else: 
                loss.backward()
                if cfg.grad_clip > 0:
                    gnorm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                else: 
                    gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in model.parameters()]), 2)
                optim.step()

            if rank == 0:
                if smooth_loss is None: smooth_loss = float(loss.item())
                else: smooth_loss = smooth_loss + 0.1*(float(loss.item()) - smooth_loss)
                # STDOUT logging
                if steps % cfg.stdout_interval == 0:
                    mb.write('steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}, peak mem: {:5.2f}GB'. \
                            format(steps, loss.item(), time.time() - start_b, torch.cuda.max_memory_allocated()/1e9))
                    mb.child.comment = 'steps : {:,d}, loss : {:4.3f}, sec/batch : {:4.3f}'. \
                            format(steps, loss.item(), time.time() - start_b)     

                # checkpointing
                if steps % cfg.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{cfg.checkpoint_path}/ckpt_{steps:08d}.pt"
                    torch.save({
                        'model_state_dict': (model.module if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1 else model).state_dict(),
                        'optim_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': (scaler.state_dict() if cfg.fp16 else None),
                        'steps': steps,
                        'epoch': epoch,
                        'cfg_yaml': OmegaConf.to_yaml(cfg)
                    }, checkpoint_path)
                    logging.info(f"Saved checkpoint to {checkpoint_path}")

                # Tensorboard summary logging
                if steps % cfg.summary_interval == 0:
                    sw.add_scalar("training/loss_smooth", smooth_loss, steps)
                    sw.add_scalar("training/loss_raw", loss.item(), steps)
                    sw.add_scalar("opt/lr", float(optim.param_groups[0]['lr']), steps)
                    sw.add_scalar('opt/grad_norm', float(gnorm), steps)

                # Validation
                if steps % cfg.validation_interval == 0 and steps != 0:
                    model.eval()
                    loss_fn.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    flat_logits = []
                    flat_lbls = []
                    with torch.no_grad():
                        for j, batch in progress_bar(enumerate(validation_loader), total=len(validation_loader), parent=mb):
                            x, y, lens = batch
                            y = y.to(device)
                            lens = lens.to(device)
                            logits = model(x.to(device), lens)
                            logits = logits.view(-1, cfg.model_cfg.mu_levels)
                            y_ = y.view(-1)
                            val_err_tot += loss_fn(logits, y_)
                            flat_logits.append(logits.cpu()) # (bs*seq_len, mu_levels)
                            flat_lbls.append(y_.cpu()) # bs*seq_len

                        val_err = val_err_tot / (j+1)
                        flat_logits = torch.cat(flat_logits, dim=0)
                        flat_lbls = torch.cat(flat_lbls, dim=0)
                        preds = flat_logits.argmax(dim=-1)
                        acc = (preds == flat_lbls).sum()/len(flat_lbls)
                        sw.add_scalar('validation/acc', float(acc), steps)
                        sw.add_scalar("validation/loss", val_err, steps)
                        mb.write(f"validation run complete at {steps:,d} steps. validation loss: {val_err:5.4f}")

                    scheduler.step(val_err)
                    model.train()
                    loss_fn.train()
                    sw.add_scalar("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9, steps)
                    sw.add_scalar("memory/max_reserved_gb", torch.cuda.max_memory_reserved()/1e9, steps)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

            steps += 1
            if steps > cfg.max_steps: 
                print("FINISHED TRAINING")
                break
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
    sw.add_hparams(flatten_cfg(cfg), metric_dict={'validation/loss': val_err}, run_name=f'run-{cfg.checkpoint_path}')
    print("Training completed!")


def main():
    print('Initializing Training Process..')
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(usage='\n' + '-'*10 + ' Default config ' + '-'*10 + '\n' + 
                            str(OmegaConf.to_yaml(OmegaConf.structured(TrainConfig))))
    a = parser.parse_known_args()
    override_cfg = OmegaConf.from_cli()
    base_cfg = OmegaConf.structured(TrainConfig)
    cfg: TrainConfig = OmegaConf.merge(base_cfg, override_cfg)
    logging.info(f"Running with config:\n {OmegaConf.to_yaml(cfg)}")

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        if cfg.distributed.n_gpus_per_node > torch.cuda.device_count():
            raise AssertionError((f" Specified n_gpus_per_node ({cfg.distributed.n_gpus_per_node})"
                                    f" must be less than or equal to cuda device count ({torch.cuda.device_count()}) "))
        with open_dict(cfg):
            cfg.batch_size_per_gpu = int(cfg.batch_size / cfg.distributed.n_gpus_per_node)
        if cfg.batch_size % cfg.distributed.n_gpus_per_node != 0:
            logging.warn(("Batch size does not evenly divide among GPUs in a node. "
                            "Likely unbalanced loads will occur."))
        logging.info(f'Batch size per GPU : {cfg.batch_size_per_gpu}')

    if cfg.distributed.n_gpus_per_node*cfg.distributed.n_nodes > 1:
       mp.spawn(train, nprocs=cfg.distributed.n_gpus_per_node, args=(cfg,))
    else:
       train(0, cfg)


if __name__ == '__main__':
    main()
