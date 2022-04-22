# twitter-analyzer-and-generator
This repository containts the code used in the [post](https://mevusas.com/%D7%98%D7%95%D7%95%D7%99%D7%98%D7%A8%20%D7%99%D7%A9%D7%A8%D7%90%D7%9C%20%D7%9B%D7%A4%D7%99%20%D7%A9%D7%9B%D7%9E%D7%94%20%D7%91%D7%95%D7%98%D7%99%D7%9D%20%D7%A1%D7%99%D7%A4%D7%A8%D7%95%20%D7%9C%D7%99) (Hebrew).

Using this code, you can scrape a twitter network based on several "kernel" accounts, cluster the resulting graph into communities, download tweets per community, fine-tune a pretrained GPT-2 model for each community and generate tweets that reperesent the style of each twitter community. every component of this pipeline can be used as a standalone module.

This work was inspired by the ["Parliament Project"](https://icc.ise.bgu.ac.il/yalla/haparlament/) (Hebrew). In addition, this project was not possible without the great hebrew language models pretrained by [Norod](https://huggingface.co/Norod78).
# Installation
`git clone git@github.com:Omri-Goldstein/twitter-analyzer-and-generator.git`

`pip install -r requirements`
# Running
Edit the lines:
`    run(get_tweets=True, train_model=True, generate_tweets=True,
        prompt='some_prompt',
        args=args)`
        
in the `main.py` file  based in your preference (command line arguments will be supported soon).

Then run `python main.py`

In order to get the tweets and build the network graph, set 'get_tweets=True'. If you have already got your texts of interets, you can set it to False (and edit the `config.py` file accordingly).

Set `train_model=True` to finetune a language model based on the training configs. If you have already saved the models, this can be set to False.

Set `generate_tweets=True` to get generate tweets per each saved language model.

# Examples
## Building and visualizing a network:
First, build a network by running:

`python main.py`

when setting:

`    run(get_tweets=True, train_model=False, generate_tweets=False,
        prompt='',
        args=args)`

This will save a .csv file with the appropriate format for [Gephi](https://gephi.org/), with which you can *manually* create these beautiful visualizations:

![image](https://user-images.githubusercontent.com/11405832/164677256-bf13570d-eed2-4462-8580-7ae4b8014692.png)


