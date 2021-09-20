# Fair and Explainable AI Models (FaX-AI)

This repository is implementation for Marrying Explainable and Fair Supervised Models (there is also a [ICML SRML 21 workshop version](https://icmlsrml2021.github.io/files/24.pdf)) and in [Supervised Learning under Discriminatory Dataset Shifts](https://arxiv.org/abs/1912.08189).
>Will add full Marrying Explainable and Fair Supervised Models arxiv link when ready.

## Training and Evaluation from Marrying Explainable and Fair Supervised Models

To train and evaluate the model(s) in the paper, run this command:

```train
python main.py
```

Dataset choice can be adjusted in lines 199-208 in main.py by modifying the path variable.

## Important Files
#### main.py
- Used to train and evaluate models for the results in our paper.

#### helper.py
- Helper functions for the main file.

#### aif_methods.py
- Compatibility layer for select methods from the [AIF360](https://github.com/Trusted-AI/AIF360) library for use with our experiments.

#### aif_metrics.py
- Compatibility layer for select metrics from the [AIF360](https://github.com/Trusted-AI/AIF360) library for use with our experiments.

## Folders
#### datasets
- Data for generating the results from the paper.
