# The whole part of the document stored in this link as the model document is too large to store in the github
https://drive.google.com/file/d/1KvfZNPKdZAihMJ4UeFF9otsMbfjEa7we/view?usp=sharing


# CLIP for Car Classification

## Code Architecture

```bash
.
├── dataset.py # load & preprocess datasets
├── example_images # store example image for demonstration
│   ├── acura tl sedan 2012.png
│   └── ...
├── models # store the pre-trained & fine-tuned models
│   ├── clip-vit-base-patch32
│   ├── finetuned_clip_model
│   ├── resnet50_model.pth
│   └── resnet50_trained.pth
├── notebooks # test functions for data loading and model loading
│   ├── test_load_datasets.ipynb
│   └── test_load_model.ipynb
├── processed_dataset # the processed dataset (we just save it for convenience)
│   ├── dataset_dict.json
│   ├── test
│   └── train
├── __pycache__ # python cache file, not important
│   ├── dataset.cpython-310.pyc
│   └── train_clip.cpython-310.pyc
├── test_clip.ipynb # Scipts for test clip model
├── train_clip.py # Scipts for fine-tuning clip model
└── train_resnet50.py # Scipts for fine-tuning resnet-50 model
```

9 directories, 21 files

## Dependencies

use the following command to install dependencies

```bash
conda create -n clip python=3.10
conda activate clip
pip install torch torchvision torchtext torchaudio transformers pandas evaluate datasets
```

## Usage

1. First, preprocess the dataset by `dataset.py`

```bash
python dataset.py
```

2. Fine-tuning the clip model

```bash
python train_clip.py
```

3. Fine-tuning the resnet50 model

```bash
python train_resnet50.py
```
