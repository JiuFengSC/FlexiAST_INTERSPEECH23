# FlexiAST: Flexibility is What AST Needs
This is the repository of the INTERSPEECH 2023 paper [FlexiAST: Flexibility is What AST Needs](https://arxiv.org/abs/2307.09286).
The core part of source code has been released, and the parts of the other experiments will be updated soon.

## Abstract
The objective of this work is to give **patch-size flexibility** to Audio Spectrogram Transformers (AST). Recent advancements in ASTs have shown superior performance in various audio-based tasks. However, the performance of standard ASTs degrates drastically when evaluated using different patch sizes from that used during training. As a result, AST models are typically re-trained to accommodate changes in patch sizes. To overcome this limitation, this paper proposes a training procedure to provide flexibility to standard AST models **without architectural changes**, allowing them to **work with various patch sizes at the inference stage** - FlexiAST. This proposed training approach simply utilizes random patch size selection and resizing of patch and positional embedding weights. Our experiments show that FlexiAST gives similar performance to standard AST models while maintaining its evaluation ability at various patch sizes on different datasets for audio classification tasks.

<div align=center>
<img width="450" alt="image" src="https://github.com/JiuFengSC/FlexiAST_INTERSPEECH23/assets/80580308/cb2373b5-e47b-40f9-a99a-c005bc7adebb">
</div>

## Released models
All the checkpoints are available by Onedrive. If the links below are expired, please contact the authors.

1. [AudioSet_DeiT384_Flexified](https://1drv.ms/u/s!AtNFLJkyZiNRgYQ2e_TKF3VQuVcAbg?e=JjdTSp)
2. [VGGSound_ViT224_Flexified](https://1drv.ms/u/s!AtNFLJkyZiNRgYQ4uUK79OCh0RJAvQ?e=I2njoh)
3. [VoxCeleb_ViT224_OnlyTime_Flexified](https://1drv.ms/u/s!AtNFLJkyZiNRgYQ3fUyrewDAikQr5A?e=cER9r3)


## How to train

Firstly, you need to manage the data according to the guidance in AST official repository.

The `*,sh` files in the `egs/{dataset}/` are the configs for training the baselines of patch sizes from 8 to 32. (Please make sure your timm version is higher than `0.8.6.dev0`). When training for baseline, the initialiaztion for AudioSet and VGGSound is ImageNet pretrained ViT, and AudioSet pretrained AST for ESC-50, SpeechCommands and VoxCeleb.

It is recommended to use the weights from pretrained baseline models as the initialization when flexifying the model. The configs in  `/egs/supvs_flexiast` are prepared for this, which both support supervised flexification (released already) and knowledge flexification (coming soon).

After getting the flexible model, we can use the `measure_flexibility.py` for evaluating the flexibility.
