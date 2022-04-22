"""Main script to run all the modules: scraping tweets from twitter, building users networks,
Fine-tuning a language model based on these tweets and generate text based on these fine-tuned models."""
import os
import sys
import time
from os import listdir
from os.path import exists
from typing import Optional

import twitter.twitter_scrapper as twitter
from config import Config
from language_model import finetune_model, generate_text


def find_suffix_filenames(path_to_dir, suffix):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def parse_args(args, cfg: Config):  # TODO parse configuration from command line
    return cfg


def run(args, get_tweets: bool = True, train_model: bool = True,
        generate_tweets: bool = True, prompt: Optional[str] = None):
    starting_time = time.time()

    cfg = Config()
    cfg = parse_args(args, cfg)
    # Build Network and Save Tweets
    if get_tweets:
        twitter.main(cfg=cfg)

    # loop over saved files to train new models
    if train_model:
        files = find_suffix_filenames(r'tweets', '.csv')
        for i, file in enumerate(files):
            if exists(os.path.join(cfg.run_config.models_path, f'model_{file}.pt')):
                continue
            print(f'Finetuning model for cluster {file}')
            finetune_model.main(cfg.run_config.tweets_path, cfg=cfg)

    # Loop over new models to generate tweets
    if generate_tweets:
        models = find_suffix_filenames(cfg.run_config.models_path, '.pt')
        if not prompt:
            prompt = input('Input your prompt to the model!')

        for i, model in enumerate(models):
            print(f'Generating text for cluster {model}')
            generate_text.main(prompt_text=prompt,
                               cfg=cfg)

    print(f'Finished After {time.time() - starting_time} seconds')


if __name__ == '__main__':
    args = sys.argv
    run(get_tweets=True, train_model=True, generate_tweets=True,
        prompt='לגבי פרשת עידית סילמן דעתי',
        args=args)
