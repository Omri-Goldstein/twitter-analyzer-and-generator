"""Fine tune a hugging face language models based on custom text."""
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 128


class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, item):
        return self.encodings[item]

    def __len__(self):
        return len(self.encodings)


def get_model_name(language: str, size: str) -> str:
    if language == 'english':
        return ''
    if language == 'hebrew' and size == 'large':
        return ''
    if language == 'hebrew' and size == 'small':
        return "Norod78/hebrew-gpt_neo-small"
    raise Exception(f'No Valid Model for language {language} and size {size}')


def load_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, pad_token_id=tokenizer.eos_token_id)
    return tokenizer, model


def train(model, models_folder: str, optimizer, tokenizer, train_loader: DataLoader, device: str,
          name: str):
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    tmp_jokes_tens = None

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch} started" + '=' * 30)
        for idx, joke in enumerate(train_loader):
            joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
            # Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if joke_tens.size()[1] > MAX_SEQ_LEN:
                continue

            # The first joke sequence in the sequence
            if not torch.is_tensor(tmp_jokes_tens):
                tmp_jokes_tens = joke_tens
                continue
            else:
                # The next joke does not fit in so we process the sequence and leave the last joke
                # as the start for next sequence
                if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                    work_jokes_tens = tmp_jokes_tens
                    tmp_jokes_tens = joke_tens
                else:
                    # Add the joke to sequence, continue and try to add more
                    tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################

            outputs = model(work_jokes_tens, labels=work_jokes_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count += 1
            if proc_seq_count == BATCH_SIZE:
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


def get_data(tweets_path: str):
    df = pd.read_csv(tweets_path)
    x_train = df['0'].tolist()
    train_dataset = TwitterDataset(x_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    return train_loader


def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def main(tweets_path: str, language: str, size: str, name: str):
    # Get data, model and device
    device = get_device()
    train_loader = get_data(tweets_path)
    tokenizer, model = load_model(get_model_name(language, size))

    # Set Optimizer
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Create a folder to save trained models
    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    # Run training loop
    train(model, models_folder, optimizer, tokenizer, train_loader, device, name)
