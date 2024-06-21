# lora-pytorch-bnet-adapted

This is a forked version of [lora-pytorch](https://github.com/fkodom/lora-pytorch) with some
minor modifications to enable seamless operation with the ByteNet implementation in
EvoDiff and certain related packages. In these packages, each conv layer takes the transpose
of its input and then returns the transpose of its output. In this lightly modified version,
we have added a quick and dirty hack to accommodate this type of layer and avoid any issues.
This forked version should only be required if you are working with ByteNet / EvoDiff;
otherwise, we recommend using lora-pytorch instead.

## Installation

```
git clone https://github.com/jlparkI/lora-pytorch-bnet-adapted
cd lora-pytorch-bnet-adapted
pip install .
```

## Usage

Is identical to lora-pytorch, with the minor difference that ByteNet layers can also
be handled.
