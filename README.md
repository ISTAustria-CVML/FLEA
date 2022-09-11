# FLEA: Provably Robust Fair Multisource Learning from Unreliable Training Data
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ISTAustria-CVML/FLEA/blob/main/LICENSE.md)

Code and example data for paper [E. Iofinova, N. Konstantinov, C. H. Lampert. "FLEA: Provably Robust Fair Multisource Learning from Unreliable Training Data"](https://arxiv.org/abs/2106.11732), Transactions of Machine Learning (TMLR), Sep 2022.

## Prerequisites (versions indicate tested setup, not minimal requirements)
Python(3.10), jax(0.3.14), numpy(1.23.1), optax(0.1.2), pandas(1.4.2), scikit-learn(1.1.1), scipy(1.8.1), folktables (0.0.11;optional), xgboost (1.6.1;optional)

## Data preprocessing
Preprocessed versions of the `adult`, `compas`, `drugs` and `germancredit` dataset are included in the `data/` directory.
The [`folktables`](https://github.com/zykls/folktables) dataset can be installed e.g. via pip.

## Running the experiments
All experiments are run using the `FLEA.py` script. Command line options are:
* `--dataset,-d`: dataset (*adult*, *compas*, *germancredit*, *drugs*, *folktables*), default: *adult*
* `--adversary,-a`: data manipulation type (see below). default: *none*
* `--nsources,-n`: total number of sources. default: 5
* `--nclean,-K`: number of clean sources. default: n//2+1
* `--nadv,-N`: number of manipulated sources. default: (n-1)//2
* `--classifier,-c`: classifier type (*linear*, *xgboost*). default: *linear*
* `--format, -F`: output format (*csv*, *json*). default: *csv*
* `--seed,-s`: random seed. default: *0*
for additional technical arguments, see the source code.

Each run of `FLEA.py` creates one split of the data with the selected
manipulations and then runes all activated ones of the following methods 
(default: all):
* `original`: train on original data before any manipulation (hypothetical)
* `clean`: merge the clean data sources and train on result (oracle)
* `all`: merge all data sources and train on result (vanilla)
* `selected`: run FLEA
* `konstantinov`: baseline *[Konstantinov et al, 2020]*
* `voting`: baseline *robust ensemble*
* `TERM`: baseline *hierarchical TERM*
* `DRO`: baseline *distributionally robust optimization*

where possible, training is performed with four different base learners:
* `fair_reg`: regularization-based fairness
* `fair_pp`: postprocessing-based fairness
* `fair_adv`: adversarial fairness
* `fair_resample`: preprocessing (resampling)-based fairness

Note: not all combinations are possible. Impossible ones will simply be skipped.

## Available data manipulation types:
* `flip#X`: flip value of attribute `X` 
* `flip#X#Y`: flip value of attributes `X` and `Y` 
* `shuffle#X`: shuffle values of attribute `X` 
* `copy#X#Y`: copy values of attribute `X` to attribute `Y`
* `resample#A`: upsample data with combinations (target=1 and protected=A) and downsample (target=1 and protected=1-A)
* `randomanchor#A`: perform anchor manipulation, see paper
* `random`: pick random data manipulation for each source

attributes `X` and `Y` can be `target` or `protected`. values for `A` can be `0` or `1`.

## Output format:

Each combination outputs four lines (for *csv*) or one json object (for *json*).
The output format is unfortunately currently rather cryptic and 
tailored towards automatic further processing rather than human readability. 
The main elements are:

* `acc`: classifier accuracy
* `auc`: classifier area under the ROC curve
* `data_pr`: fraction of positive labels in dataset
* `cls_pr`: fraction of positive labels in classifier output
* `tpr`, `tnr`: classifier true positive/true negative ratio

further entries include these statistics also evaluated 
separately for each protected group (`group-0`, `group-1`)
and their difference (`delta`). The *demographic parity* violation 
can be read off from the difference of the `cls_pr` values between 
the protected groups.

## Visualization of results

The script `plot_results.py` creates bar plots of the results 
as in the manuscript and outputs tables in LaTeX format. 
It expects results from runs with all adversaries for random 
seeds 0,...,9 in a subdirectory `./results`

### BibTex:
```
@article{iofinova-tmlr2022,
  author = {Eugenia Iofinova and Nikola Konstantinov and Christoph H. Lampert},
  title = {{FLEA}: Provably Robust Fair Multisource Learning from Unreliable Training Data},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year = 2022
}
```
