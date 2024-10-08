[![CC BY license](https://img.shields.io/badge/License-CC%20BY-lightgray.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.11-gold.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.0-pumpkin.svg)](https://pytorch.org/get-started/previous-versions/#v200)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-purple)](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending)

# 👀 What?
This repository contains code for using the $d_{HM}$ evaluation method proposed in:  
**[*Not (yet) the whole story*: Evaluating Visual Storytelling Requires More than Measuring Coherence, Grounding, and Repetition](https://arxiv.org/pdf/2407.04559)**&mdash;In proceedings of EMNLP 2024 (*Findings*).

**Note:** Despite being proposed specifically for visual storytelling, this method is generalizable and can be extended to any task involving model-generated outputs with corresponding references.

# 🤔 Why?
$d_{HM}$ enables human-centric evaluation of model-generated stories along different dimensions important for visual story generation.

# 🤖 How?
$d_{HM}$ combines three reference-free evaluation metrics&mdash;GROOViST[^1] (for visual grounding), RoViST-C[^2] (for coherence), and RoViST-NR[^2] (for non-redundancy/repetition)&mdash;by computing the average of absolute metric-level deviations between human stories and corresponding model generations.

[^1]: https://aclanthology.org/2023.emnlp-main.202/
[^2]: https://aclanthology.org/2022.findings-naacl.206/

## Setup
Install python (e.g., version `3.11`) and other dependencies provided under [requirements.txt](./requirements.txt), e.g., using:  
`pip install -r requirements.txt`

## Step 0: Generate stories
For generating stories using the models and settings proposed in this work, refer to [this documentation](./generate-stories/README.md).

## Step 1A: Compute metric-level scores for human stories
For computing visual grounding scores (`G`), checkout the [GROOViST](https://github.com/akskuchi/groovist/) repository.

For computing coherence (`C`) and repetition (`R`) scores, use the following utility adapted from [RoViST](https://github.com/usydnlp/rovist). E.g.,  
`python evaluate/eval_C_R.py -i ./data/stories/vist/gt_test.json -o ./data/scores/vist/gt_test`  

**Note 1:** Download the pre-trained ALBERT model from [here](https://drive.google.com/file/d/1-ATRk6AQyKGNDZHkqrKkpjiY6jfbK9NS/view?usp=sharing) and place it under the [`data/`](./data/) folder.

**Note 2:** Requirements differ&mdash;checkout the [evaluate/requirements](./evaluate/requirements.txt) file.

## Step 1B: Compute metric-level scores for model-generated stories
Similar to Step `1A`.

## Step 2: Evaluate using $d_{HM}$
For obtaining aggregate $d_{HM}$ values along with corresponding metric-level distances ($d_{HM}^G, d_{HM}^C, d_{HM}^R$), use the following utility. E.g.,  
`python dHM.py -d VIST`

---
🔗 If you find this work useful, please consider citing it:
```
@inproceedings{
   EMNLP 2024 Findings (to appear) 
}
```
