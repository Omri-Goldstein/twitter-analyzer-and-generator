"""Generate text from a trained language model."""
import os

import numpy as np
import torch

from config import Config
from language_model.finetune_model import load_model, get_model_name


def generate_text(prompt_text: str, model, tokenizer, cfg: Config):
    max_len = cfg.generation_config.max_len
    sample_output_num = cfg.generation_config.sample_output_num
    seed = cfg.generation_config.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = 0 if torch.cuda.is_available() is False else torch.cuda.device_count()

    print(f"device: {device}, n_gpu: {n_gpu}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    model.to(device)

    encoded_prompt = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt")

    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    print("input_ids = " + str(input_ids))

    if input_ids != None:
        max_len += len(encoded_prompt[0])
        if max_len > 2048:
            max_len = 2048

    print("Updated max_len = " + str(max_len))

    stop_token = "<|endoftext|>"
    new_lines = "\n\n\n"

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_len,
        top_k=cfg.generation_config.top_k,
        top_p=cfg.generation_config.top_p,
        num_return_sequences=sample_output_num
    )

    print(100 * '-' + "\n\t\tOutput\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token else None]

        # Remove all text after 3 newlines
        text = text[: text.find(new_lines) if new_lines else None]

        print("\n{}: {}".format(i, text))
        print("\n" + 100 * '-')


def main(prompt_text: str, model_checkpoint: str, cfg: Config = Config()):
    tokenizer, model = load_model(get_model_name(cfg))
    model.load_state_dict(torch.load(os.path.join(cfg.run_config.models_path, model_checkpoint)))
    model.eval()
    generate_text(prompt_text, model, tokenizer, cfg)
