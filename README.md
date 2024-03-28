# Title generator

**WIP**

The goal of this project is to have an (L)LM to generate titles for conversations. This is meant to be used, for example, as a title-generator for [open-webui](https://github.com/open-webui/open-webui) as a custom [ollama](https://github.com/ollama/ollama) model.

For that, a language model was fine-tuned to always output a 3-5 words title summarizing the input.

## Model Information
The model was fine-tuned on a dataset of conversations, assembled from [Puffin](https://huggingface.co/datasets/LDJnr/Puffin) and [chatalpaca-20k](https://github.com/cascip/ChatAlpaca/tree/main/data) (see `experiments/chatalpaca.ipynb` for the why). The final dataset is available at `data/dataset.jsonl`.

I decided to take only the first message and use `gpt-3.5-turbo` to generate synthetic titles (see `src/generate.py`). 
