import time
from os import listdir

import twitter.twitter_scrapper as twitter
from language_model import finetune_model, generate_text


def find_suffix_filenames(path_to_dir, suffix):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def run():
    starting_time = time.time()

    # Build Network and Save Tweets
    twitter.main()

    # loop over saved files to train new models
    files = find_suffix_filenames(r'tweets', '.csv')
    for i, file in enumerate(files):
        finetune_model.main(file, language='hebrew', size='small', name=f'model_{i}')

    # Loop over new models to generate tweets
    models = find_suffix_filenames(r'trained_models', '.pt')
    for i, model in enumerate(models):
        print(f'generating text for cluster {i}')
        generate_text.main('trained_models/' + model,
                           language='hebrew',
                           size='small',
                           prompt_text='יש בדיוק בעיה אחת במדינה והיא')

    print(f'Finished After {time.time() - starting_time} seconds')


if __name__ == '__main__':
    run()
