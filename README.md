# LCS-CBM: Logic Concept Subspace-aware Bottleneck Model for Interpretable Medical Image Diagnosis

Official code repository for **LCS-CBM**, a concept-based framework for interpretable medical image diagnosis.

## Overview

Conventional concept bottleneck models typically use a single global concept-to-label mapping. LCS-CBM instead learns multiple **logic concept subspaces** and uses **concept-conditioned assignment** to adaptively combine their predictions. Each subspace models differentiable logical interactions over continuous concept probabilities. For an individual prediction, the top-ranked subspace is used to extract a compact concept-level explanation.

Key components include:

- Differentiable AND/OR reasoning over continuous concept probabilities.
- A bank of logic concept subspaces with distinct learnable parameters.
- Concept-conditioned routing and probability-level prediction aggregation.
- Subspace load balancing and compact rule extraction.

The method is evaluated on **Fitzpatrick17k (F17k), PBC, MiniDDSM, IDRiD, and NCT**.

## Installation

Clone the repository and create a Python 3.10 environment:

```bash
git clone https://github.com/lifxiyou/LCS-CBM.git
cd LCS-CBM
conda create -n lcs python=3.10
conda activate lcs
```

Install a compatible version of **PyTorch** and **torchvision** for your CUDA environment (see the [official PyTorch instructions](https://pytorch.org/get-started/locally/)), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The currently provided `requirements.txt` is a minimal dependency list. Depending on your environment, additional packages used by the source code (such as `PyYAML`) may be required. Please verify the environment before running experiments.

## Dataset Preparation

Download the datasets and their required concept annotations from the corresponding sources:

| Dataset | Domain | Source |
| --- | --- | --- |
| **Fitzpatrick17k (F17k)** | Dermatology | [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) and [SkinCon](https://skincon-dataset.github.io/) |
| **PBC** | Peripheral blood-cell microscopy | [PBC images](https://data.mendeley.com/) and the [WBCAtt annotation resource](https://github.com/ybgao22/CRL) |
| **MiniDDSM** | Mammography | [Mini-DDSM dataset](https://ardisdataset.github.io/MiniDDSM/) |
| **IDRiD** | Retinal fundus images | [IDRiD](https://idrid.grand-challenge.org/Data/) |
| **NCT** | Histopathology | [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) |

For each dataset, place the raw images and corresponding CSV metadata/concept annotations under its configured data directory:

```text
[data_dir]/
├── raw_data/
│   └── ...                 # Images
└── meta_data/
    └── ...                 # Metadata and concept-annotation CSV files
```

Update `data.data_dir` in the relevant configuration file in `configs/` to point to the actual dataset directory on your machine. **Do not use the absolute paths from the original training server without changing them.**

### Five-fold split for F17k

Once the F17k images and the required `fitzpatrick17k.csv` and `skincon.csv` files are in place, run:

```bash
python -m data.split_dataset
```

This creates a `5_fold_splits.json` file under the F17k `meta_data/` directory.

> **Important:** The released `data/split_dataset.py` currently contains a hard-coded dataset path. Set its `data_dir` variable to your local F17k directory before running this command. The five-fold splitting instruction applies to **F17k**; other datasets use their corresponding preparation and evaluation protocols.

For MiniDDSM, IDRiD, and NCT, follow the dataset-specific concept-supervision protocol described in the paper and the relevant dataset-processing code. The repository does not yet provide one universal preprocessing command for all five datasets.

## Repository Structure

```text
LCS-CBM/
├── configs/             # Experiment configurations
├── data/                # Data preparation and splitting utilities
├── dataset/             # Dataset loaders
├── models/              # Model definitions and logic components
├── callbacks.py         # Training callbacks
├── main.py              # Training and evaluation entry point
├── metrics.py           # Evaluation metrics
├── test.py              # Evaluation utilities
├── utils.py             # Shared utilities
├── requirements.txt     # Python dependencies
└── README.md
```

## Training and Evaluation

Experiments are configured through YAML files in `configs/`. The main entry point accepts the `--configs` argument and configuration overrides.

The following are the **intended command patterns** for the existing skin and blood-cell configurations:

```bash
# F17k, example fold 0
python main.py --configs configs/skin.yaml --data.fold 0

# PBC
python main.py --configs configs/wbc.yaml
```

For the five F17k folds, change `--data.fold` from `0` to `4`.

> **Before running:** Review the selected configuration and the `main.py` imports. The current public `main.py` references `models.CNSCRL_MoE`, which is not included in the uploaded model files, and its default model initialization should be checked against the final LCS-CBM implementation. These commands are therefore **not yet verified as an end-to-end reproduction script**. Dataset-specific commands for MiniDDSM, IDRiD, and NCT should be added after validation.

## Citation

If you use this repository, please cite our paper:

> **LCS-CBM: Logic Concept Subspace-aware Bottleneck Model for Interpretable Medical Image Diagnosis**  
> Fan Li, Rui Yan, Chunjiang Wang, Yujia Li, Wenbo Hu, Kun Zhang, and Shaohua Kevin Zhou.

The final BibTeX entry will be added when the publication metadata is available.

## Acknowledgment

This work was supported by the National Natural Science Foundation of China (62502490, 62271465); the Natural Science Foundation of Jiangsu Province (BK20250496, BK20255001); the Sanming Project of Medicine in Shenzhen (SZSM202411033); the National Key R&D Program of China (2025YFC3408300); and the Suzhou Basic Research Program (SYG202338).

We acknowledge the creators of the datasets and concept annotations used in this work. Please follow each dataset's license and citation requirements.
