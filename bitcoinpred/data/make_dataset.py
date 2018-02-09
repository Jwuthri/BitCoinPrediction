# -*- coding: utf-8 -*-
"""
@author: JulienWuthrich
"""
import os
import json
import requests
import logging
import time
import datetime

import numpy as np

from textblob import TextBlob

from bitcoinpred.config.settings import raw_data_path
from bitcoinpred.config.settings import logger
from bitcoinpred.config.logger import logged
from bitcoinpred.data.settings import *
from bitcoinpred.features.twitter import TweetCollector


class BitCoinCollector(object):
    """Module to collect the blockchain values."""

    def __init__(self, filename):
        """Initialize a `MakeDataset` object.

            Arg:
                filename (str): path of the file
        """
        self.filename = filename
        self.coinmarket = "https://api.coindesk.com/v1/bpi/historical/close.json?start={}&end={}"

    @logged(level=logging.INFO, name=logger)
    def collect(self):
        """Collect the data from api.coinmarketcap.com each minutes.

            Return:
                data (csv): save the data collected in a csv file
        """
        file = open(self.filename, "a")
        path1 = os.path.join(raw_data_path, "tweets1.csv")
        path2 = os.path.join(raw_data_path, "reddits1.csv")
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, 100)]
        for date in date_list:
            prev_hour = 0
            for hour in [7, 15, 23]:
                until = date.replace(hour=hour, minute=59, second=59)
                since = date.replace(hour=prev_hour, minute=59, second=59)
                since = since - datetime.timedelta(days=1)
                prev_hour = hour
                tweet_sentiment, reddit_sentiment = SentimentCollector(path1, path2, since, until).collect()
                tim = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
                logger.log(logging.DEBUG, "== Collecting BitCoin information at {} ==".format(tim))
                data = requests.get(self.coinmarket.format(since.strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d"))).json().get("bpi").get(since.strftime("%Y-%m-%d"))
                file.write(str(data) + ";" + tweet_sentiment + ";" + reddit_sentiment + ";" + str(until) + "\n")
                file.flush()
                time.sleep(30)


class SentimentCollector(object):
    """Module to collect the data from Twitter and Reddit values."""

    def __init__(self, tweet_file, reddit_file, since, until):
        """Initialize a `SentimentCollector` object.

            Args:
                tweet_file (str): path of the file for the tweets
                reddit_file (str): path of the file for the reddit posts
        """
        self.tweet_file = tweet_file
        self.reddit_file = reddit_file
        self.since = since
        self.until = until
        self.archive_reddit = "http://archive.org/wayback/available?url=reddit.com/r/bitcoin&timestamp="
        self.reddit_api = "http://api.idolondemand.com/1/api/sync/analyzesentiment/v1?apikey={}&url=".format(reddit_api_key)

    def get_tweet_sentiment(self):
        """Get the general sentiment of the tweets.

            Return:
                tweet_sentiment (float): mean of the sentiment
        """
        file = open(self.tweet_file, "a")
        logger.log(logging.DEBUG, "== Writing tweets ==")
        since = self.since.strftime("%Y-%m-%dT%H:%M:%S")
        until = self.until.strftime("%Y-%m-%dT%H:%M:%S")
        tweets = TweetCollector(query="bitcoin, price, crypto, blockchain", count=25, since=since, until=until).collect()
        tweet_polarity = list()
        for tweet in tweets:
            text = tweet.text.encode("ascii", errors="ignore").decode()
            logger.log(logging.INFO, "Tweet {} ... ".format(text[:10]))
            file.write(text + '\n')
            analysis = TextBlob(tweet.text)
            tweet_polarity.append(analysis.sentiment.polarity)

        return np.mean(tweet_polarity)

    def get_reddit_sentiment(self):
        """Get the general sentiment of the reddit posts.

            Return:
                reddit_sentiment (float): mean of the sentiment
        """
        file = open(self.reddit_file, "a")
        date = self.until.strftime("%Y%m%d%H%M%S")
        reddit = requests.get(self.archive_reddit + date)
        if reddit.status_code == 200:
            data1 = json.loads(reddit.text)
            try:
                archive_url = data1['archived_snapshots']['closest']['url']
                reddit_posts = requests.get(self.reddit_api + archive_url)
                if reddit_posts.status_code == 200:
                    reddit_posts_sentiment = json.loads(reddit_posts.text)
                    for key in ["positive", "negative"]:
                        for post in reddit_posts_sentiment[key]:
                            for col in reddit_cols:
                                val = str(post[col])
                                val = val.encode("ascii", errors="ignore").decode()
                                file.write(val + ";")
                            file.write(str(date) + ";" + key + "\n")

                    return reddit_posts_sentiment['aggregate']['score']
            except:
                pass
        return None

    @logged(level=logging.INFO, name=logger)
    def collect(self):
        """Collect the data from api.twitter.

            Return:
                data (csv): save the data collected in a csv file
        """
        tim = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        logger.log(logging.DEBUG, "== Collecting new Tweets at {} ==".format(tim))
        tweet_sentiment = self.get_tweet_sentiment()
        logger.log(logging.DEBUG, "== Collecting Reddit new at {} ==".format(tim))
        reddit_sentiment = self.get_reddit_sentiment()

        return str(tweet_sentiment), str(reddit_sentiment)


if __name__ == '__main__':
    path = os.path.join(raw_data_path, "bitcoin1.csv")
    BitCoinCollector(path).collect()
