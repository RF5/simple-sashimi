"""
SaShiMi backbone.

Use this backbone in your own models. You'll also need to copy over the
standalone S4 layer, which can be found at 
`state-spaces/src/models/sequence/ss/standalone/s4.py`.

It's Raw! Audio Generation with State-Space Models
Karan Goel, Albert Gu, Chris Donahue, Christopher Re. 

Adapted from https://github.com/HazyResearch/state-spaces/blob/diffwave/sashimi/sashimi.py
"""

import warnings
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from einops import rearrange
from fastprogress.fastprogress import progress_bar


from sashimi.s4 import S4, LinearActivation
from sashimi.config import AutoregressiveConfig, DiffusionConfig


def swish(x):
    return x * torch.sigmoid(x)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Taken from https://github.com/philsyn/DiffWave-Vocoder

    Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):
                                    dimensionality of the embedding space for discrete diffusion steps

    Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(diffusion_steps.device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)

    return diffusion_step_embed

class Conv(nn.Module):
    """
    Dilated conv layer with kaiming_normal initialization
    from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    """
    Conv1x1 layer with zero initialization 
    From https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
    """
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x, **kwargs):
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x, **kwargs):
        x = self.linear(x)
        
        if self.causal:
            # Shift to ensure causality
            x = F.pad(x[..., :-1], (1, 0))

        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state


class FFBlock(nn.Module):

    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = LinearActivation(
            d_model, 
            d_model * expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand,
            d_model, 
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = nn.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def forward(self, x, **kwargs):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        return self.ff(x.unsqueeze(-1)).squeeze(-1), state


class ResidualBlock(nn.Module):

    def __init__(
        self, 
        d_model, 
        layer,
        dropout=0.0,
    ):
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, **kwargs):
        """
        Input x is shape (B, d_input, L)
        """
        z = x
        
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)

        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x, None

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)

        # Residual connection
        x = z + x

        return x, state


class DiffWaveS4Block(nn.Module):
    """
    Modified DiffWave block that uses S4.

    Taken from https://github.com/philsyn/DiffWave-Vocoder
    """
    def __init__(self,
            d_model, 
            diffusion_step_embed_dim_out=512,
            unconditional=False,
            mel_upsample=[16, 16],
            l_max=16000,
            postact=None,
            tie_state=False
        ):
        super().__init__()
        self.d_model = d_model

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.d_model)

        self.layer = S4(
            d_model, 
            bidirectional=True,
            hurwitz=True, # use the Hurwitz parameterization for stability
            tie_state=tie_state, # tie SSM parameters across d_state in the S4 layer
            trainable={
                'dt': True,
                'A': True,
                'P': True,
                'B': True,
            }, # train all internal S4 parameters
            l_max=l_max,
            postact=postact
        )
        self.norm = nn.LayerNorm(d_model)

        self.unconditional = unconditional
        if not self.unconditional:
            # add mel spectrogram upsampler and conditioner conv1x1 layer
            self.upsample_conv2d = torch.nn.ModuleList()
            for s in mel_upsample:
                conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
                conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
                torch.nn.init.kaiming_normal_(conv_trans2d.weight)
                self.upsample_conv2d.append(conv_trans2d)
            self.mel_conv = Conv(80, self.d_model, kernel_size=1)  # 80 is mel bands

    def forward(self, x, diffusion_step_embed, mel_spec=None):
        y = x
        B, C, L = x.shape
        assert C == self.d_model

        y = self.norm(y.transpose(-1, -2)).transpose(-1, -2)

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        y = y + part_t.unsqueeze(-1)
        
        # S4 layer
        y, _ = self.layer(y)

        # add mel spectrogram as (local) conditioner
        if mel_spec is not None:
            assert not self.unconditional
            # Upsample spectrogram to size of audio
            mel_spec = torch.unsqueeze(mel_spec, dim=1)
            mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4)
            mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4)
            mel_spec = torch.squeeze(mel_spec, dim=1)

            assert(mel_spec.size(2) >= L)
            if mel_spec.size(2) > L:
                mel_spec = mel_spec[:, :, :L]

            mel_spec = self.mel_conv(mel_spec)
            y = y + mel_spec

        # Residual
        y = x + y

        return y, None

class Sashimi(nn.Module):
    def __init__(
        self,
        d_model=64, 
        n_layers=8, 
        pool=[4, 4], 
        expand=2, 
        ff=2, 
        bidirectional=False,
        glu=True,
        unet=False,
        diffwave=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        SaShiMi model backbone. 

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level. 
                We use 8 layers for our experiments, although we found that increasing layers even further generally 
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels. 
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models 
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling. 
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            diffwave: switch to DiffWave model with SaShiMi backbone. We use this variant for our diffusion
                models. Note that S4 is bidirectional by default in this variant, and we recommend switching 
                on the `unet` argument as well. Additional kwargs for 
                    - `diffusion_step_embed_dim_in` (default 128)
                    - `diffusion_step_embed_dim_mid` (default 512)
                    - `diffusion_step_embed_dim_out` (default 512)
                    - `unconditional` (default False)
                    - `mel_upsample` (default [16, 16])
                can be passed in to control the SaShiMi diffusion model.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet
        self.diffwave = diffwave

        # Bidirectional S4 layers are always used in DiffWave
        bidirectional = bidirectional or diffwave 

        if self.diffwave and not self.unet: 
            warnings.warn("DiffWave is not recommended without UNet. Consider using UNet instead.")

        def s4_block(dim, kwargs):
            layer = S4(
                d_model=dim, 
                d_state=64,
                bidirectional=bidirectional,
                postact='glu' if glu else None,
                dropout=dropout,
                transposed=True,
                hurwitz=True, # use the Hurwitz parameterization for stability
                tie_state=True, # tie SSM parameters across d_state in the S4 layer
                # trainable={
                    # 'dt': True,
                    # 'A': True,
                    # 'P': True,
                    # 'B': True,
                # }, # train all internal S4 parameters
                trainable=kwargs['trainable'],
                lr=kwargs['lr'],
                l_max=kwargs['l_max'],                    
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        def ff_block(dim):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
            )

        if diffwave:
            # Setup for DiffWave SaShiMi model
            # Borrows code from https://github.com/philsyn/DiffWave-Vocoder

            self.diffusion_step_embed_dim_in = kwargs.get('diffusion_step_embed_dim_in', 128)
            self.diffusion_step_embed_dim_mid = kwargs.get('diffusion_step_embed_dim_mid', 512)
            self.diffusion_step_embed_dim_out = kwargs.get('diffusion_step_embed_dim_out', 512)
            in_channels = 1
            out_channels = 1
            
            # Initial conv1x1 with relu
            self.init_conv = nn.Sequential(Conv(in_channels, d_model, kernel_size=1), nn.ReLU())

            # the shared two fc layers for diffusion step embedding
            self.fc_t1 = nn.Linear(self.diffusion_step_embed_dim_in, self.diffusion_step_embed_dim_mid)
            self.fc_t2 = nn.Linear(self.diffusion_step_embed_dim_mid, self.diffusion_step_embed_dim_out)

            # Final conv1x1 -> relu -> zeroconv1x1
            self.final_conv = nn.Sequential(
                Conv(d_model, d_model, kernel_size=1),
                nn.ReLU(),
                ZeroConv1d(d_model, out_channels),
            )

            def s4_block(dim, kwargs):
                return DiffWaveS4Block(
                    d_model=dim,
                    diffusion_step_embed_dim_out=self.diffusion_step_embed_dim_out,
                    unconditional=kwargs.get('unconditional', True),
                    mel_upsample=kwargs.get('mel_upsample', [16, 16]),
                    l_max=kwargs['l_max'],
                    postact='glu' if glu else None,
                    tie_state=kwargs['tie_state']
                )

        # Down blocks
        d_layers = []
        for p in pool:
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    d_layers.append(s4_block(H, kwargs))
                    if ff > 0: d_layers.append(ff_block(H))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        
        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, kwargs))
            if ff > 0: c_layers.append(ff_block(H))
        
        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal=not bidirectional))

            for _ in range(n_layers):
                block.append(s4_block(H, kwargs))
                if ff > 0: block.append(ff_block(H))

            u_layers.append(nn.ModuleList(block))
        
        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        assert H == d_model

    def forward(self, x, state=None, mel_spec=None):
        """
        input: (batch, length, d_input)
        output: (batch, length, d_output)
        """
        if self.diffwave:
            audio, diffusion_steps = x
            x = audio
            # BLD -> BDL
            x = x.transpose(1, 2)

            x = self.init_conv(x)

            diffusion_step_embed = calc_diffusion_step_embedding(
                diffusion_steps, 
                self.diffusion_step_embed_dim_in,
            )
            diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
            diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

            # Additional kwargs to pass onto the DiffWaveS4Block
            layer_kwargs = dict(diffusion_step_embed=diffusion_step_embed, mel_spec=mel_spec)
        else:
            # BLD -> BDL
            x = x.transpose(1, 2)

            # No additional kwargs to pass onto the S4 & FF blocks
            layer_kwargs = dict()

        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            x, _ = layer(x, **layer_kwargs)
            outputs.append(x)

        # Center block
        for layer in self.c_layers:
            x, _ = layer(x, **layer_kwargs)
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    x, _ = layer(x, **layer_kwargs)
                    x = x + outputs.pop() # skip connection
            else:
                for layer in block:
                    x, _ = layer(x, **layer_kwargs)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        if self.diffwave:
            x = self.final_conv(x.transpose(1, 2)).transpose(1, 2)

        return x, None # required to return a state

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped//3:]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state

    def setup_rnn(self, mode='dense'):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.

        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%. 
                `linear` should be faster theoretically but is slow in practice since it 
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ['dense', 'diagonal', 'linear']
        for module in self.modules():
            if hasattr(module, 'setup_step'): module.setup_step(mode)


class SashimiAR(nn.Module):

    supports_nll: bool = True

    def __init__(self, cfg: AutoregressiveConfig) -> None:
        super().__init__()

        self.sashimi = Sashimi(**cfg)
        self.mu_embedder = nn.Embedding(cfg.mu_levels, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.mu_levels)
        self.cfg = cfg

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """ Accepts mu-law encoded batch `x` (bs, seq_len), int64, returns float logits of same shape """
        x = self.mu_embedder(x) # (bs, seq_len, dim)
        y, _ = self.sashimi(x)
        logits = self.head(y) # (bs, seq_len, mu_levels)
        return logits

    @torch.inference_mode()
    def unconditional_generate(self, N: int, nucleus_p=1.0, progress=True, mb=None) -> Tensor:
        """ Generate `N` audio samples, returning a tensor of shape (N, 16000) """
        mu_tfm = torchaudio.transforms.MuLawEncoding(self.cfg.mu_levels)
        mu_detfm = torchaudio.transforms.MuLawDecoding(self.cfg.mu_levels)

        bs = N
        device = next(self.parameters()).device
        x = torch.zeros(bs, 1)
        x = mu_tfm(x).to(device)
        x = self.mu_embedder(x)
        x = x.squeeze(1)
        
        self.sashimi.setup_rnn() # setup S4 layers in recurrent mode
        # alternately, use sashimi.setup_rnn('diagonal') for a speedup
        # Run recurrence
        ys = []
        state = self.sashimi.default_state(*x.shape[:1], device=device)
        if progress: pb = progress_bar(range(16000), parent=mb)
        else: pb = range(16000)

        for i in pb:
            y_, state = self.sashimi.step(x, state)
            logits = self.head(y_)
            
            ## Special sampling methods come in here.
            
            # nucleus sampling 
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1).detach()
            sorted_probs, sorted_inds = torch.sort(probs, dim=-1, descending=True)
            cs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cs > nucleus_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for b in range(bs):
                indices_to_remove = sorted_inds[b, sorted_indices_to_remove[b]]
                probs[b, indices_to_remove] = 0.0

            # Renormalize:
            probs = probs / probs.sum(dim=-1)[:, None]
            # Sample next sample:
            ix = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_input = ix
            
            if progress: pb.comment = f"sampled token: {int(next_input[0])}"
            x = self.mu_embedder(next_input)
            
            ys.append(next_input.cpu())

        ys = torch.stack(ys, dim=1) # ys.shape == x.shape
        audio = mu_detfm(ys)
        return audio

    @torch.inference_mode()
    def nll(self, wavs: Tensor) -> Tensor:
        """ Gets likelihood for `wavs` (N, 16000), returning neg log likelihoods (N,) """
        mu_tfm = torchaudio.transforms.MuLawEncoding(self.cfg.mu_levels)

        bs = wavs.shape[0]
        device = next(self.parameters()).device
        wavs_ = F.pad(wavs, (1, 0), value=0)
        x = mu_tfm(wavs_).to(device)
        logits = self.forward(x[:, :-1], None)
        # we predict the next token, so we must offset by one
        targ_inds = x[:, 1:]
        logits = torch.gather(logits, -1, targ_inds[..., None]).squeeze(-1)
        logprobs = F.log_softmax(logits, dim=-1) # (N, 16000)
        likelihood = logprobs.sum(dim=-1) # (N,)
        nll = -likelihood
        per_sam_nll = nll/wavs.shape[-1]
        per_sam_nll_b2 = per_sam_nll/math.log(2, math.e)
        print(per_sam_nll_b2)
        raise AssertionError()
        return nll
        

class SashimiDiffWave(nn.Module):

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()

        self.sashimi = Sashimi(**cfg)
        print(f"Initialized DiffWave Sashimi with {sum([p.numel() for p in self.parameters()]):,d} parameters.")

    def forward(self, x:Tensor, diffusion_steps: Tensor, spectrogram=None) -> Tensor:
        """ Accepts raw waveforms in range [-1, 1] `x` of shape (bs, seq_len) and
        `diffusion_steps` of shape (bs,). `spectrogram` is unused and should not be given. 
        Only done for compatibility with diffwave training code.
        """
        # internally we must add extra dimensions
        x = x[..., None] # (bs, seq_len, 1)
        diffusion_steps = diffusion_steps[..., None] # (bs, 1)
        y, _ = self.sashimi((x, diffusion_steps)) # y is (bs, seq_len, 1)
        y = y.squeeze(-1)
        return y


if __name__ == '__main__':
    from fastprogress.fastprogress import progress_bar

    print("Testing autoregressive method")
    # Example: SaShiMi for autoregressive modeling
    model = Sashimi(n_layers=2).cuda()
    # Print parameter count
    print(f"Total parameters = {sum(p.numel() for p in model.parameters()):,d}")

    model.eval()

    with torch.no_grad():
        # Forward in convolutional mode: used for training SaShiMi
        x = torch.randn(3, 10240, 64).cuda()
        y, _ = model(x)
        
        # Setup the SaShiMi RNN
        model.setup_rnn('diagonal')

        # Forward in recurrent mode: used for autoregressive generation at inference time
        ys = []
        state = model.default_state(*x.shape[:1], device='cuda')
        for i in progress_bar(range(10240)):
            y_, state = model.step(x[:, i], state)
            ys.append(y_.detach().cpu())
        
        ys = torch.stack(ys, dim=1)

        print(y.shape, ys.shape)

    print("Testing diffusion method")
    # Example: SaShiMi for diffusion modeling
    model = Sashimi(n_layers=2, diffwave=True, unet=True).cuda()
    # Print parameter count
    print(sum(p.numel() for p in model.parameters()))

    model.eval()

    with torch.no_grad():
        # Forward (only) in convolutional mode
        x = torch.randn(3, 10240, 1).cuda() # (bs, seq_len, 1)
        steps = torch.randint(0, 4, (3, 1)).cuda()
        y, _ = model((x, steps))
        print(y.shape)
