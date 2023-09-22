# FlexiAST: Flexibility is What AST Needs
This is the repository of the INTERSPEECH 2023 paper [FlexiAST: Flexibility is What AST Needs](https://arxiv.org/abs/2307.09286).
The core part of source code has been released, and the parts of the other experiments will be updated soon.

## Abstract
The objective of this work is to give **patch-size flexibility** to Audio Spectrogram Transformers (AST). Recent advancements in ASTs have shown superior performance in various audio-based tasks. However, the performance of standard ASTs degrates drastically when evaluated using different patch sizes from that used during training. As a result, AST models are typically re-trained to accommodate changes in patch sizes. To overcome this limitation, this paper proposes a training procedure to provide flexibility to standard AST models **without architectural changes**, allowing them to **work with various patch sizes at the inference stage** - FlexiAST. This proposed training approach simply utilizes random patch size selection and resizing of patch and positional embedding weights. Our experiments show that FlexiAST gives similar performance to standard AST models while maintaining its evaluation ability at various patch sizes on different datasets for audio classification tasks.





<div align=center>
<img width="450" alt="image" src="https://github.com/JiuFengSC/FlexiAST_INTERSPEECH23/assets/80580308/cb2373b5-e47b-40f9-a99a-c005bc7adebb">
</div>
