"""Scrape tweets from twitter, build a network based on retweets, find communities and save their tweets."""
import os
from typing import List, Dict, Optional

import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tqdm as tqdm
import tweepy
from tweepy import OAuthHandler
from os.path import exists

from config import Config


class TwitterScraper:
    def __init__(self, api):
        self.api = api

    def get_all_tweets_of_user(self, screen_name: str, num_tweets: int = 4000, only_text: bool = False) -> List:
        try:
            tweets = self.api.user_timeline(screen_name=screen_name,
                                            # 200 is the maximum allowed count
                                            count=200,
                                            include_rts=True,
                                            # Necessary to keep full_text
                                            # otherwise only the first 140 words are extracted
                                            tweet_mode='extended'
                                            )
        except:
            return []
        all_tweets = []
        all_tweets.extend(tweets)
        oldest_id = tweets[-1].id
        while True:
            try:
                tweets = self.api.user_timeline(screen_name=screen_name,
                                                # 200 is the maximum allowed count
                                                count=200,
                                                include_rts=True,
                                                max_id=oldest_id - 1,
                                                # Necessary to keep full_text
                                                # otherwise only the first 140 words are extracted
                                                tweet_mode='extended'
                                                )
            except:
                break
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

    def get_all_tweets_from_names_list(self, names_list: List[str], num_tweets: int = 4000) -> List:
        tweets = []
        for name in tqdm.tqdm(names_list):
            tweets += self.get_all_tweets_of_user(name, only_text=True, num_tweets=num_tweets)

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

    @staticmethod
    def save_tweets_to_csv(tweets: List, name: str):
        pd.Series(tweets).to_csv(name)


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

    def _find_communities(self):
        self.partition = community.best_partition(self.graph)
        self.communities = set(self.partition.values())

    def _get_tweets_of_community(self, cluster: int, num_tweets: int) -> List[str]:
        tweets = []
        for key, value in tqdm.tqdm(self.partition.items()):
            if value == cluster:
                tweets += self.scrapper.get_all_tweets_of_user(key, only_text=True, num_tweets=num_tweets)

        return tweets

    def save_tweets_from_per_community_from_graph(self):
        models_folder = "tweets"
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)
        for cluster in self.communities:
            print(f'Getting tweets from community {cluster}')
            tweets = self._get_tweets_of_community(cluster=cluster, num_tweets=200)
            self.scrapper.save_tweets_to_csv(tweets, f'tweets/community_{cluster}.csv')

    def build_graph_and_save_tweets_per_community(self, users: List[str], save_tweets_per_community: bool = True):
        self.build_graph(users)
        self.save_to_gephi('gephi_graph.csv')
        self._find_communities()
        if save_tweets_per_community:
            self.save_tweets_from_per_community_from_graph()

    def save_to_gephi(self, filename: str):
        df = pd.DataFrame(self.graph.edges(data=True))
        df.columns = ['Source', 'Target', 'Weight']
        nx.write_edgelist(self.graph, filename, data=False)
        df['Weight'] = df.Weight.apply(lambda x: x['weight'])
        df.to_csv(filename, index=False)


def save_tweets_per_community_from_gephi_df(scraper: TwitterScraper, df: pd.DataFrame, clustering_threshold: int = 0):
    df = df[df.clustering > clustering_threshold]
    for community in df.modularity_class.unique():
        if exists( f'tweets/community_{community}.csv'):
            continue
        print(f'Saving tweets for community {community}')
        users_list = df[df.modularity_class == community].Label.tolist()
        tweets = scraper.get_all_tweets_from_names_list(users_list, num_tweets=400)
        scraper.save_tweets_to_csv(tweets, f'tweets/community_{community}.csv')


def save_all_tweets_from_users_list(filename: str, scrapper: TwitterScraper, users_list: List[str]):
    tweets = scrapper.get_all_tweets_from_names_list(users_list)
    scrapper.save_tweets_to_csv(tweets, f'tweets/community_{filename}.csv')


def get_twitter_api():
    api_key = os.getenv('API_KEY')
    api_secret_key = os.getenv('API_SECRET_KEY')
    auth = OAuthHandler(
        api_key,
        api_secret_key,

    )
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def main(users_list: Optional[List] = None,
         filename: Optional[str] = None,
         cfg: Config = Config()):

    build_graph = cfg.run_config.build_graph
    save_tweets_per_community = cfg.run_config.save_tweets_per_community
    gephi_file = cfg.run_config.gephi_file
    api = get_twitter_api()
    users = cfg.users

    scrapper = TwitterScraper(api)

    if build_graph:
        graph = GraphBuilder(scrapper)
        graph.build_graph_and_save_tweets_per_community(users, save_tweets_per_community)
    elif users_list:
        save_all_tweets_from_users_list(filename, scrapper, users_list)
    else:
        df = pd.read_csv(gephi_file)
        save_tweets_per_community_from_gephi_df(scrapper, df)
