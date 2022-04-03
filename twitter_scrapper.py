import os
from typing import List, Dict

import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tqdm as tqdm
import tweepy
from tweepy import OAuthHandler

from .config import Config


class TwitterScraper:
    def __init__(self, api):
        self.api = api

    def get_all_tweets_of_user(self, screen_name: str, num_tweets: int = 1000, only_text: bool = False) -> List:

        tweets = self.api.user_timeline(screen_name=screen_name,
                                        # 200 is the maximum allowed count
                                        count=200,
                                        include_rts=True,
                                        # Necessary to keep full_text
                                        # otherwise only the first 140 words are extracted
                                        tweet_mode='extended'
                                        )
        all_tweets = []
        all_tweets.extend(tweets)
        oldest_id = tweets[-1].id
        while True:
            tweets = self.api.user_timeline(screen_name=screen_name,
                                            # 200 is the maximum allowed count
                                            count=200,
                                            include_rts=True,
                                            max_id=oldest_id - 1,
                                            # Necessary to keep full_text
                                            # otherwise only the first 140 words are extracted
                                            tweet_mode='extended'
                                            )
            if len(tweets) == 0:
                break
            if len(all_tweets) > num_tweets:
                break
            oldest_id = tweets[-1].id
            all_tweets.extend(tweets)
            # print('N of tweets downloaded till now {}'.format(len(all_tweets)))
        if only_text:
            all_tweets = [tweet._json['full_text'] for tweet in all_tweets]

        return all_tweets

    def get_all_tweets_from_names_list(self, names_list: List[str]) -> List:
        for name in tqdm.tqdm_notebook(names_list):
            tweets = self.get_all_tweets_of_user(name, only_text=True)

        return tweets

    @staticmethod
    def _get_retweeted_id(tweet) -> str:
        return tweet._json['full_text'].split('@')[1].split(':')[0]

    def get_retweeted_dict(self, tweets: List) -> Dict:
        retweeted_dict = dict()

        for tweet in tweets:
            if tweet._json['full_text'][:2] == 'RT':
                retweeted = self._get_retweeted_id(tweet)
                if retweeted in retweeted_dict:
                    retweeted_dict[retweeted] += 1
                else:
                    retweeted_dict[retweeted] = 1
        return retweeted_dict


class GraphBuilder:
    def __init__(self, scrapper: TwitterScraper):
        self.scrapper = scrapper
        self.graph = nx.Graph()
        self.net = list()
        self.partition = dict()
        self.communities = set()

    def build_graph(self, screen_names: List[str], threshold=1, plot=False):
        print('Building Graph')
        net = []
        weights = []
        for screen_name in tqdm.tqdm(screen_names):
            tweets = self.scrapper.get_all_tweets_of_user(screen_name)
            retweets_dict = self.scrapper.get_retweeted_dict(tweets)
            for retweet in retweets_dict:
                if retweets_dict[retweet] > threshold:
                    net.append((screen_name, retweet))
                    weights.append(retweets_dict[retweet])

        G = nx.Graph()
        for edge, weight in zip(net, weights):
            G.add_edge(edge[0], edge[1], weight=weight)

        to_be_removed = [x for x in G.nodes() if G.degree(x) <= 1]

        for x in to_be_removed:
            G.remove_node(x)

        if plot:
            plt.figure(figsize=(15, 15))
            nx.draw_networkx(G)
            plt.show()
        self.graph = G
        self.net = net

    def find_communities(self):
        self.partition = community.best_partition(self.graph)
        self.communities = set(self.partition.values())

    def get_tweets_of_community(self, cluster: int, num_tweets: int) -> List[str]:
        tweets = []
        for key, value in tqdm.tqdm(self.partition.items()):
            if value == cluster:
                tweets += self.scrapper.get_all_tweets_of_user(key, only_text=True, num_tweets=num_tweets)

        return tweets

    @staticmethod
    def save_tweets_to_csv(tweets: List, name: str):
        pd.Series(tweets).to_csv(name)

    @staticmethod
    def save_to_gephi():
        pass


def build_graph_and_save_tweets_per_community(scrapper, users):
    graph_builder = GraphBuilder(scrapper)
    graph_builder.build_graph(users)
    graph_builder.find_communities()
    models_folder = "tweets"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)
    for cluster in graph_builder.communities:
        print(f'Getting tweets from community {cluster}')
        tweets = graph_builder.get_tweets_of_community(cluster=cluster, num_tweets=200)
        graph_builder.save_tweets_to_csv(tweets, f'tweets/community_{cluster}.csv')


def save_all_tweets_from_users_list(filename: str, scrapper: TwitterScraper, users_list: List[str]):
    tweets = scrapper.get_all_tweets_from_names_list(users_list)
    GraphBuilder.save_tweets_to_csv(tweets, f'tweets/community_{filename}.csv')


def get_twitter_api():
    api_key = os.getenv('API_KEY')
    api_secret_key = os.getenv('API_SECRET_KEY')
    auth = OAuthHandler(
        api_key,
        api_secret_key,

    )
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def main(build_graph: bool = True, users_list: List = [], filename: str = None):
    params = Config()
    api = get_twitter_api()
    users = params.users

    scrapper = TwitterScraper(api)

    if build_graph:
        build_graph_and_save_tweets_per_community(scrapper, users)
    else:
        save_all_tweets_from_users_list(filename, scrapper, users_list)
