>📋  A template README.md for code accompanying a Machine Learning paper

# Fair and Explainable AI Models (FaX-AI)

This repository is the official implementation of the methods used in Marrying Explainable and Fair Supervised Models and in [Supervised Learning under Discriminatory Dataset Shifts](https://arxiv.org/abs/1912.08189).
>Insert influence arxiv link when ready.

Our methods include: the input influence optimization methods using SHAP and average treatment effect (ATE), the marginal interventional mixture (MIM), and the optimal interventional mixture (OIM).

NOTE: When we refer to our paper in this repository we refer to Marrying Explainable and Fair Supervised Models. To produce the results from [Supervised Learning under Discriminatory Dataset Shifts](https://arxiv.org/abs/1912.08189) please refer to this [repository](https://github.com/social-info-lab/discrimination-prevention/tree/master/src).

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

1. For our methods, [Python 3.8](https://www.python.org/downloads/release/python-380/), [numpy, scipy](https://www.scipy.org/scipylib/download.html), [matplotlib](http://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/), and [PyTorch](https://pytorch.org/get-started/locally/).
2. For other methods and to produce the paper's results, all of the above plus [AIF360](https://github.com/Trusted-AI/AIF360) and [Pandas](https://pandas.pydata.org/).

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Using Our Methods
All of our methods can be found within the file "fair_mass.py". The following classes have the implementations: Optimization, OIM, and MIM.

We include other implementations support our methods such as logistic regression, the underlying estimator for our methods, and the influence measures of SHAP and ATE.

Datasets must be in numpy-type arrays. Additionally, the protected attribute separated from other features when using our methods and be fed in as the 'Z' parameter. 'X' is the remaining features and 'Y' is the target.

Note that our methods train when they are initialized.

## Training and Evaluation from Marrying Explainable and Fair Supervised Models

To train and evaluate the model(s) in the paper, run this command:

```train
python main.py
```

Dataset choice can be adjusted in lines 199-208 by modifying the path variable.

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Files in this Repo

#### fair_mass.py
- Main file containing the implementation of our methods and input influence measures.

### Files for Experiments
#### main.py
- Used to train and evaluate models for the results in our paper.

#### helper.py
- Helper functions for the main file.

#### aif_methods.py
- Compatibility layer for select methods from the [AIF360](https://github.com/Trusted-AI/AIF360) library for use with our experiments.

#### aif_metrics.py
- Compatibility layer for select metrics from the [AIF360](https://github.com/Trusted-AI/AIF360) library for use with our experiments.

### Folders
#### datasets
- Data for generating the results from the paper.

#### alternate implementations
- Contains implementations and experiments for older versions of the paper. 
