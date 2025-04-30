# Language-Identifier

This repository contains a language identification (LID) script built using Hugging Face Transformers and datasets. The script fine-tunes a pretrained transformer model for sequence classification on the OpenLID-v2 dataset (or custom datasets) to identify the language of given text samples. It supports training, evaluation, and hyperparameter tuning with Weights & Biases (wandb) integration.

## Setup

Install dependencies using:

```bash
pip install -r requirements.txt
```

Clone the repository and create a `.env` file with your API keys:
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

Prepare datasets:

- If `--skip_loading` is `True` (default), place `train.csv`, `val.csv`, and `test.csv` with columns `text` and `label` in the `data/` directory.
- If `--skip_loading` is `False`, the script downloads and preprocesses the OpenLID-v2 dataset automatically.


## Usage

Run the script with optional arguments:

- `--model_id`: Hugging Face model ID (e.g., `"FacebookAI/xlm-roberta-large"`)

- `--skip_train`: Set to `False` to train the model; set to `True` to run inference only.

- `--skip_loading`: Set to `True` to load data from pre-saved CSV files; otherwise, loads the OpenLID-v2 dataset from Hugging Face.

- `--n_epoch`: Number of training epochs (default: `2`)


### Training Mode
```
python classifier.py --model_id "FacebookAI/xlm-roberta-large" --skip_train False --skip_loading True --n_epoch 2
```

### Inference Mode
```
python classifier.py --model_id "FacebookAI/xlm-roberta-large" --skip_train True --skip_loading True --n_epoch 2
```

