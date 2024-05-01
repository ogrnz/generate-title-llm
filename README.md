# Title generator

The goal of this project is to have an (L)LM to generate titles for conversations. This is meant to be used, for example, as a title-generator for [open-webui](https://github.com/open-webui/open-webui) as a custom [ollama](https://github.com/ollama/ollama) model.

For that, a language model was fine-tuned to always output a 3-5 words title summarizing the input.

## Model Information

The dataset used to fine-tune the model is available on Hugging Face: https://huggingface.co/datasets/ogrnz/chat-titles.

## Setup

If you only want to do inference:
```bash
# 1. Clone the repo
git clone https://github.com/ogrnz/generate-title-llm.git
cd generate-title-llm

# 2. Create a python virtual env and activate it

# 3. Install transformers
pip install transformers

# 4. Use it!
python src/infer.py --message "I have troubles with my self-hosted docker setup. A traefik2 reverse proxy handles requests from the internet to my different services."
```

If you want to use the full project (generate new synthetic dataset, fine-tune another model, modify a script...):
```bash
# 1. Clone the repo
git clone https://github.com/ogrnz/generate-title-llm.git
cd generate-title-llm

# 2. Create a python virtual env and activate it

# 3. Install transformers
pip install -r requirements.txt
```

## Usage

### Basic

For basic inference: 
```bash
python src/infer.py --message "I have troubles with my self-hosted docker setup. A traefik2 reverse proxy handles requests from the internet to my different services."
```

You can also pass a JSONL file with the `--file_path` argument.

### `src/finetune.py`

```bash
# Sane default
python src/finetune.py 

# Custom
python src/finetune.py --checkpoint "google-t5/t5-small" --hf-data "ogrnz/chat-titles" --output-dir results
```

Custom training arguments (for example to not use fp16) can be written in a json file and loaded via the `--training-config` flag:

```json
{
    "fp16": false,
}
```

```bash
# Custom
python src/finetune.py --training-config training_config.json
```

### `src/infer.py`

Script used to perform inference.

### `src/dataset.py`

Script used to generate the synthetic dataset.