# ECG Semantic Integrator (ESI): A Foundation ECG Model

This repository contains the implementation of the pretraining phase for the paper "ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text".

https://openreview.net/forum?id=giEbq8Khcf

@Article{yu2024ecg,

  title={ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text},

  author={Yu, Han and Guo, Peikun and Sano, Akane},
  
  journal={Transactions on Machine Learning Research (TMLR)},
  
  year={2024}

}


## Overview

ESI is a novel approach to improve electrocardiogram (ECG) analysis using deep learning techniques. It consists of two main components:

1. **Cardio Query Assistant (CQA)**: A retrieval-augmented generation (RAG) pipeline that generates detailed textual descriptions for ECG data (see `rag/rag_llamaindex.py` as an LLAMAINDEX version; the generated waveform descriptions for MIMIC-IV-ECG are in './rag/mimic_waveform_dict.json').
2. **ECG Semantics Integrator (ESI)**: A multimodal contrastive pretraining framework that combines ECG signals with textual descriptions to learn robust representations.

## Key Features

- Utilizes both contrastive and captioning losses for pretraining
- Employs a 1D modified ConvNext v2 architecture for ECG encoding
- Uses BioLinkBERT (pretrained on biomedical texts) for text encoding
- Supports various ECG signal encoders (ConvNeXtV2 variants and XResNet1D)

## Requirements

- Python 3.8+
- PyTorch 1.7+
- Transformers

## Installation

```bash
git clone https://github.com/your-username/esi-ecg-model.git
cd esi-ecg-model
```

## Usage

To train the ESI model:

```bash
python main.py --meta_path /path/to/meta_data.csv --file_path /path/to/ecg/files --batch_size 48 --epochs 50 --signal_encoder convnextv2_base --text_encoder michiyasunaga/BioLinkBERT-base
```

## Configuration

The `main.py` script accepts various command-line arguments to customize the training process. Some key parameters include:

- `--signal_encoder`: Choose from 'convnextv2_base', 'convnextv2_atto', 'convnextv2_nano', 'convnextv2_tiny', 'convnextv2_large', or 'xresnet1d101'
- `--text_encoder`: Specify the pretrained text encoder (default: 'michiyasunaga/BioLinkBERT-base')
- `--dim`: Model dimension (default: 768)
- `--lr`: Learning rate (default: 5e-5)
- `--batch_size`: Batch size per GPU (default: 8)

For a full list of configuration options, refer to the argument parser in `main.py`.

## Training Details

- Optimizer: AdamW
- Learning rate schedule: Warm-up for 5 epochs, decay by 0.1 every 10 epochs
- Total epochs: 30
- Hardware: 4 Nvidia A100 GPUs (adjust `--gpu_ids` as needed)

## Evaluation

The pretrained ESI model can be evaluated on downstream tasks such as arrhythmia diagnosis and ECG-based user identification. Refer to the paper for detailed evaluation protocols and results.

## Citation

If you use this code or find our work helpful, please cite our paper:

```
@article{yu2024ecg,
  title={ECG Semantic Integrator (ESI): A Foundation ECG Model Pretrained with LLM-Enhanced Cardiological Text},
  author={Yu, Han and Guo, Peikun and Sano, Akane},
  journal={arXiv preprint arXiv:2405.19366},
  year={2024}
}
```

## Acknowledgements
This code is adapted from [COCA-pytorch](https://github.com/lucidrains/coca-pytorch).
The evaluation for arrhythmia diagnosis uses codes at [PTB-XL Benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking).
