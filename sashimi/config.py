from dataclasses import MISSING, dataclass, field
from typing import List, Tuple, Union
import numpy as np


def fix(blah): return field(default_factory=lambda: blah)


@dataclass
class AutoregressiveConfig:
    """ SC09 autoregressive parameters. 
    There are discrepancies between the paper descriptions of hyperparams
    and the repo's description of hyperparams (e.g. paper states B is not optimized, but 
    repo config states B is optimized.)

    So, I use the original paper hyperparameters were available, and use the state-spaces repo
    hyperparameters as a fallback for those parameters the paper does not state.       
    """
    mu_levels: int = 256
    # Model settings 
    glu: bool = True
    d_model: int = 64
    n_layers: int = 8
    pool: List[int] = fix([4, 4])
    expand: int = 2
    ff: int = 2
    bidirectional: bool = False
    unet: bool = False
    diffwave: bool = False
    dropout: float = 0.0
    # From paper:
    # For S4 parameters, we only train Λ and C with the recommended learning rate of
    # 0.001, and freeze all other parameters for simplicity (including pp∗, B, dt)
    trainable: dict = fix({
        'dt': True,
        'A': True,
        # C is always trained. 
        'P': True,
        'B': True,
    })
    lr: float = 0.001
    l_max: int = 16000


@dataclass
class DiffusionConfig:

    model_dir: str = MISSING
    data_dir: str = MISSING

    # Model settings 
    glu: bool = True
    l_max: int = 16000
    tie_state: bool = False
    d_model: int = 128
    n_layers: int = 6
    pool: List[int] = fix([4, 4])
    expand: int = 2
    ff: int = 2
    unet: bool = True
    diffwave: bool = True
    dropout: float = 0.0
    # optimization settings
    batch_size: int = 16
    learning_rate: float = 2e-4
    max_grad_norm: Union[int, None] = None
    # diffusion settings
    unconditional: bool = True
    sample_rate: int = 16000
    audio_len: int = 16000

    noise_schedule: List[float] = fix(np.linspace(1e-4, 0.02, 200).tolist())
    inference_noise_schedule: List[float] = fix([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])

    max_steps: int = 800000 # 800k
    fp16: bool = False
    seed: int = 123

    # And I quote:
    # "All optimization and diffusion hyperparameters were kept the same, with the 
    # exception that we manually decayed the learning rate of the large SaShiMi model 
    # at 500K steps as it had saturated and the model had already caught up 
    # to the best DiffWave model."
    # ... What did they decay it to? lol very reproducible :)
    # I am making a best guess as being the same as the plateau decay factor of 0.2
    # used in the autoregressive type models. 
    decay_factor_at_500k: float = 0.2
