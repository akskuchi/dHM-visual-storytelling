# What?
This repository contains code for using the $\text{d}_{HM}$ evaluation method proposed in:  
[*Not (yet) the whole story*: Evaluating Visual Storytelling Requires More than Measuring Coherence, Grounding, and Repetition](https://arxiv.org/pdf/2407.04559)

# Why?
$\text{d}_{HM}$ enables human-centric evaluation of model-generated stories along different dimensions important for visual story generation.

# How?
$\text{d}_{HM}$ combines three reference-free evaluation metrics&mdash;GROOViST[^1] (for visual grounding), RoViST-C[^2] (for coherence), and RoViST-NR[^2] (for non-redundancy).

[^1]: https://aclanthology.org/2023.emnlp-main.202/
[^2]: https://aclanthology.org/2022.findings-naacl.206/

## Setup

Install python (e.g., version `3.11`) and other dependencies provided in [requirements.txt](./requirements.txt), e.g., using:  
`pip install -r requirements.txt`

## Step 0: Compute metric-level scores for human-written stories

For computing visual grounding scores, checkout the [GROOViST](https://github.com/akskuchi/groovist/) repository.

For computing coherence and non-redundancy scores, use the following utility:  
`python score_c_r.py --help`

## Step 1: Compute metric-level scores for model-generated stories

Similar to Step `0`.

## Step 2: Compute $\text{d}_{HM}$

For obtaining aggregate $\text{d}_{HM}$ values along with corresponding metric-level distances ($\text{d}_{HM}^G, \text{d}_{HM}^C, \text{d}_{HM}^R$), use the following utility:  
`python dHM.py --help`
