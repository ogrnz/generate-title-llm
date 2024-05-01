# Title generator

The goal of this project is to have an (L)LM to generate titles for conversations. This is meant to be used, for example, as a title-generator for [open-webui](https://github.com/open-webui/open-webui) as a custom [ollama](https://github.com/ollama/ollama) model.

For that, a language model was fine-tuned to always output a 3-5 words title summarizing the input.

## Model Information

The dataset used to fine-tune the model is available on Hugging Face: https://huggingface.co/datasets/ogrnz/chat-titles.

## Usage
