# Fair and Explainable AI Models (FaX-AI)

This repository is the official implementation of the methods used in Marrying Explainable and Fair Supervised Models (there is also a [ICML SRML 21 workshop version](https://icmlsrml2021.github.io/files/24.pdf)) and in [Supervised Learning under Discriminatory Dataset Shifts](https://arxiv.org/abs/1912.08189).
>Will add full Marrying Explainable and Fair Supervised Models arxiv link when ready.

Our methods include: the input influence optimization methods using SHAP and average treatment effect (ATE), the marginal interventional mixture (MIM), and the optimal interventional mixture (OIM).

NOTE: When we refer to our paper in this repository we refer to Marrying Explainable and Fair Supervised Models. To produce the results from [Supervised Learning under Discriminatory Dataset Shifts](https://arxiv.org/abs/1912.08189) please refer to this [repository](https://github.com/social-info-lab/discrimination-prevention/tree/master/src).

## Requirements

To install requirements:

1. For our methods, [Python >=3.6](https://www.python.org/downloads/release/python-370/), [numpy, scipy](https://www.scipy.org/scipylib/download.html), [matplotlib](http://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/), and [PyTorch](https://pytorch.org/get-started/locally/).
2. For the examples and other methods, all of the above plus [AIF360](https://github.com/Trusted-AI/AIF360) and [Pandas](https://pandas.pydata.org/).

## Using Our Methods
All of our methods can be found within the file "fair_mass.py". The following classes have the implementations: Optimization, OIM, and MIM.

We include other implementations support our methods such as logistic regression, the underlying estimator for our methods, and the influence measures of SHAP and ATE.

Datasets must be in numpy-type arrays. Additionally, the protected attribute separated from other features when using our methods and be fed in as the 'Z' parameter. 'X' is the remaining features and 'Y' is the target.

Note that our methods train when they are initialized.

## Important Files in this Repo

#### fair_mass.py
- Main file containing the implementation of our methods and input influence measures.

### Example files
#### simple_example.py
- Shows how to use our Optimization method using randomly generated NumPy arrays.

#### compas_example.py
- Shows how to load and preprocess the COMPAS data from the [AIF360](https://github.com/Trusted-AI/AIF360) library and use it our Optimization.

### Folders
#### example
- Contains code for the Marrying Explainable and Fair Supervised Models results.
