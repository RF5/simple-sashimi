from dataclasses import dataclass, field
from typing import List, Tuple, Union

# from omegaconf import MISSING, OmegaConf, open_dict
# from omegaconf.dictconfig import DictConfig
# from omegaconf.listconfig import ListConfig

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
        'dt': False,
        'A': True,
        # C is always trained. 
        'P': False,
        'B': False,
    })
    lr: float = 0.001
    l_max: int = 16000


@dataclass
class DiffusionConfig:
    pass
