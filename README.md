# simple-sashimi
A simple-to-use standalone Pytorch implementation of the autoregressive SaShiMi unconditional speech synthesis model.

**Original paper**: ["It's Raw! Audio Generation with State-Space Models"](https://arxiv.org/abs/2202.09729)

**Original paper authors**: Karan Goel, Albert Gu, Chris Donahue, Christopher Ré

## Models
I train and provide two models -- both trained on the **Google Speech Commands SC09 dataset** (consisting of 1s, 16kHz spoken digits 'zero' through 'nine'). 
The two models are detailed:

| Model | Params | Training iters | Checkpoint |
| -- | --: | :--: | :--: |
| Autoregressive Sashimi | 4.8M | 1.1M | [weights link](https://github.com/RF5/simple-sashimi/releases/download/v1.0/ckpt_01100000.pt) , [hosted tensorboard logs](https://tensorboard.dev/experiment/S3QB9BAVQYOOEA09VSndVQ/#scalars) |
| DiffWave Sashimi | 23.5M | 800k | TODO |

### Quickstart 
Both models support torchhub, and assuming you have `torch`, `torchaudio`, `einops`, `opt_einsum`, and `scipy` installed in your python environment, you can simply run:

```python
import torch
# load model
model = torch.hub.load('RF5/simple-sashimi', 'sashimi_ar_sc09', device='cuda') # use cpu if no gpu
# generate e.g. 4 utterances with no nucleus sampling (i.e. direct greedy sampling).
audio = model.unconditional_generate(4, nucleus_p=1.0)
# audio is now (4, 16000) tensor of 4 one-second utterances.

```

That's it!
Alternatively, `model.forward()` accepts mu-law encoded batch `x` of shape (bs, seq_len), int64, and returns float logits of same shape over the mu-law encoded bins if you wish to implement your own custom sampling methods.

### Differences to the original paper
The [original work](https://arxiv.org/abs/2202.09729) leaves out several pieces of information necessary to reproduce the work, and the provided [source code](https://github.com/HazyResearch/state-spaces/) does not include details on the diffusion experiments. 

Furthermore, the source code conflicts with the paper in many cases (e.g. paper states `B` is not optimized, but repo config states `B` is optimized within S4 layers). 

Left with such a quandry, I made my best attempt at using parameters from the paper where specified, and falling back to parameters specified in the source code when the paper did not specify. Additionally, for specific parameters that the author raised in issues on the original repo, I set the training settings to their recommended values over what the paper specifies to obtain best performance. 

**TL;DR**: the models trained are as similar as possible to the original paper, if anything they are slightly better than those presented in the paper. 
The full model config for both the autoregressive and diffusion Sashimi models are given in `sashimi/config.py`. 

## Requirements

**For inference**:  `torch`, `torchaudio`, `einops`, `opt_einsum`, and `scipy`

**For training**: the same as inference, but also `fastprogress`, `omegaconf`, tensorboard plugin for torch, `pandas`, and optionally the CUDA kernel specified below. 

### Using the cauchy kernel speedup
To speed and **reduce memory usage** up cauchy kernel computation, you must compile the CUDA kernel operations from [the original repo](https://github.com/HazyResearch/state-spaces/tree/main#cauchy-kernel). 
To do this make sure you hae a gcc version newer than 4.9 but less than 9.0, otherwise nvcc or torch throws a fit. 
Without this, the S4 layer appears to use an exorbitant amount of memory for the number of parameters used. The official implementation of the S4 layer taken from [the original repo](https://github.com/HazyResearch/state-spaces/) even appears to slightly leak memory when not using these additional kernel or pykeops dependencies,  so if you are training this model, I highly recommend installing the CUDA kernel.

Note: I have noticed that without using the cauchy kernel, the final model performance is sometimes less than what it is with the cauchy kernel. I suspect there may be a discrepancy in the reference CPU implementation and custom CUDA kernel implementation. In any event, I recommend using the cauchy kernel. 

## Training
To train the autoregressive model, clone the repo, and run `python train.py --help` to see available arguments. 
The defaults are the ones used to train the pretrained autoregressive model, and you can override any setting with `<setting>=<option>`. For example:

```bash
python train.py train_csv=splits/sc09-train.csv valid_csv=splits/sc09-valid.csv checkpoint_path=runs/run1/ validation_interval=2500
```

You can generate splits with `python split_data.py --help` to see options to get train/valid/test csv files for just SC09 or for the full google speech commands dataset. 

To train the diffwave diffusion model, please see my [diffwave repo](https://github.com/RF5/DiffWave-unconditional).

# Acknowledgements
This repo uses code adapted from several previous works:

1. The [state-spaces repo of the original SaShiMi authors](https://github.com/HazyResearch/state-spaces/)
2. Some diffusion training and inference code from the [DiffWave author's official implementation](https://github.com/lmnt-com/diffwave)

Thanks to the original authors for the cool work you have done in developing SaShiMi.
