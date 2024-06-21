## Evodiff FT

This variant of EvoDiff has been slightly altered to be amenable
to fine-tuning with or without LoRA.


### Installation

Create a new conda environment with python v3.8.5.

```
conda create --name evodiff python=3.8.5
```

In that new environment, install EvoDiff:
```
pip install evodiff
```

Next, clone this repository and also a repository that provides
a simple LoRA wrapper for the EvoDiff model. The wrapper will
need to be installed as illustrated below.

```
git clone https://github.com/jlparkI/lora-pytorch-bnet-adapted
cd lora-pytorch-bnet-adapted
pip install .

cd ..
git clone https://github.com/jlparkI/evodiff_ft
```

### Fine-tuning

To fine-tune EvoDiff, you'll create a shell script in the ``evodiff_ft``
folder -- an example shell script is already included. This shell
script will supply some key arguments to the ``fine_tune.py`` file
and call it. Run this on GPU (on GPU-3 preferably). These arguments are:

```
--config_fpath The filepath to a config file. You can generally use the ones
    under config. Under config there is one json file for the 38M parameter model
    and one for the 640M parameter model -- use the one that's appropriate for
    what you are trying to tune.
--out_fpath A filepath to a folder where the fine-tuned model and intermediate
    results (checkpoints, loss on each epoch) will be saved.
--train_fpath A filepath to a file containing the sequences you would like to fine-tune
    on. This file should have "Sequence" in the first line then all remaining lines are
    sequences.
--valid_fpath A filepath to a file containing a validation set youy would like to score
    after each epoch during training. This should have the same format as the training file.
--checkpoint_freq How many minutes to save a checkpoint at.
--large_model Supply this if you want to use the 640M parameter model; otherwise do not supply
    this and the 38M parameter model will be used.
--LoRA Supply an integer to indicate the rank for LoRA. If you do not supply this argument LoRA
    will not be used and the full model will be fine-tuned.
```

After fine-tuning, the state dict for the updated model will be saved to your output path, where you can load
it for further use, under the name ``"FINAL_MODEL.pt"``.

If you want to change the learning rate, number of epochs etc that will be used during fine-
tuning, change the appropriate config file under config. Note that batch size is fairly
important -- if you set this too high you may encounter an out of memory error. Using larger
batches however can definitely speed up fine-tuning.
