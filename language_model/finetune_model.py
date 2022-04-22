"""Fine tune a hugging face language models based on custom text."""
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config


class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, item):
        return self.encodings[item]

    def __len__(self):
        return len(self.encodings)


def get_model_name(cfg: Config) -> str:
    if cfg.network_config.language == 'english':
        return cfg.network_config.english_model
    if cfg.network_config.language == 'hebrew' and cfg.network_config.model_size == 'large':
        return cfg.network_config.large_hebrew_model
    if cfg.network_config.language == 'hebrew' and cfg.network_config.model_size == 'small':
        return cfg.network_config.small_hebrew_model
    raise Exception(f'No Valid Model for language {cfg.network_config.language} and size '
                    f'{cfg.network_config.model_size}')


def load_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, pad_token_id=tokenizer.eos_token_id)
    return tokenizer, model


def train(model, models_folder: str, optimizer, tokenizer,
          train_loader: DataLoader, device: str, name: str,
          cfg: Config):
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    tmp_tweets_tens = None

    for epoch in range(cfg.network_config.epochs):
        print(f"EPOCH {epoch} started" + '=' * 30)
        for idx, tweet in enumerate(train_loader):
            tweet_tens = torch.tensor(tokenizer.encode(tweet[0])).unsqueeze(0).to(device)
            # Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if tweet_tens.size()[1] > cfg.network_config.max_seq_len:
                continue

            # The first joke sequence in the sequence
            if not torch.is_tensor(tmp_tweets_tens):
                tmp_tweets_tens = tweet_tens
                continue
            else:
                # The next joke does not fit in so we process the sequence and leave the last joke
                # as the start for next sequence
                if tmp_tweets_tens.size()[1] + tweet_tens.size()[1] > cfg.network_config.max_seq_len:
                    work_tweet_tens = tmp_tweets_tens
                    tmp_tweets_tens = tweet_tens
                else:
                    # Add the joke to sequence, continue and try to add more
                    tmp_tweets_tens = torch.cat([tmp_tweets_tens, tweet_tens[:, 1:]], dim=1)
                    continue
            # Sequence ready, process it trough the model ##################

            outputs = model(work_tweet_tens, labels=work_tweet_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count += 1
            if proc_seq_count == cfg.network_config.batch_size:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
            if batch_count == 1:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(models_folder, f"{name}.pt"))


def get_data(tweets_path: str, cfg: Config):
    df = pd.read_csv(tweets_path)
    x_train = df['0'].tolist()
    train_dataset = TwitterDataset(x_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.network_config.batch_size)
    return train_loader


def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def main(name: str, cfg: Config = Config()):
    # Get data, model and device
    device = get_device()
    train_loader = get_data(cfg.run_config.tweets_path, cfg)
    tokenizer, model = load_model(get_model_name(cfg))

    # Set Optimizer
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.network_config.learning_rate)

    # Create a folder to save trained models
    models_folder = cfg.run_config.models_path
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    # Run training loop
    train(model, models_folder, optimizer, tokenizer, train_loader, device, name, cfg)
