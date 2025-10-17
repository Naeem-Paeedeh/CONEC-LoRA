# CONNEC-LoRA

The official repository for the **CONEC-LoRA** algorithm as described in the **Continual Knowledge Consolidation LORA for Domain Incremental Learning** paper.

## Preparing the datasets

The datasets used in the paper can be downloaded as described in this section. Please note that the "data_path" setting in the .json files in the [exps](/exps) must be set to the directories that are used for each dataset.

### CDDB

Please refer to the [project page](https://coral79.github.io/CDDB_web/) of the [CDDB repository](https://github.com/Coral79/CDDB). You can download the dataset from the following [Google Drive](https://drive.google.com/file/d/1NgB8ytBMFBFwyXJQvdVT_yek1EaaEHrg/view?usp=sharing)

### Office-Home

The following link is the [Office-Home's project page](https://www.hemanthdv.org/officeHomeDataset.html).

### CORe50

The CORe50 dataset can be downloaded from the [Project page](https://vlomonaco.github.io/core50/index.html) of the [CORe50 repository](https://github.com/vlomonaco/core50).

### DomainNet

The DomainNet dataset can be downloaded from [the project page of the Moment Matching for Multi-Source Domain Adaptation](https://ai.bu.edu/M3SDA/).

## Python and libraries

We performed the experiments with the following versions of Python and the main libraries:

- `python==3.9.23`
- `torch==2.0.1`
- `torchvision==0.15.2`
- `numpy==1.26.4`
- `timm==0.6.12`
- `scikit-learn==1.6.1`
- `python-box==7.3.2`
- `scipy==1.13.1`
- `einops==0.8.1`

## Running the experiments

You can run the following scirpts [cddb.sh](cddb.sh), [office-home](office-home.sh), [core50.sh](core50), and [domainnet.sh](domainnet).

## Ablation studies

You can run the [ablation_study.sh](ablation_study.sh) for the ablation studies.

## Robustness analysis

The robustness analysis can be performed by running the [robustness_analysis.sh](robustness_analysis.sh).

## Acknowledgement

We thank the authors of the [CL-LoRA](https://github.com/JiangpengHe/CL-LoRA) and [DCE](https://github.com/Lain810/DCE), [SOYO](https://github.com/QWangCV/SOYO), [FLOWER]([FLOWER](https://github.com/anwarmaxsum/FLOWER)), and [S3C](https://github.com/JAYATEJAK/S3C) for providing their code, upon which our code is built.
