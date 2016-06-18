import pandas as pd
import numpy as np
import tweepy as tp
import file_paths as fp
import json
import csv


consumer_key = 'HpbA4sFAg5ac0dXG4Sm9hflLD'
consumer_secret = 'lTYzPwYzZewGn9e0HdZgj9vtb8B9TfHvGNF7vYoUp5e2xDT31f'
access_token = '1107345421-9DUxSHtoGphHsabgWnfiUtIKmeLKy8AzCyLWVje'
access_token_secret = '1MCkGFPIB4IcM9oqhTwPnlwX5o2AXh6LA5LrcCvuAPIXr'

auth = tp.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tp.API(auth)
#%% Creating a Streaming Session

class TPStreamListener(tp.StreamListener):

    def __init__(self, tweet_limit=10, tweet_file_name = './bf_tweets.csv'):
        self.tweet_limit = tweet_limit
        self.tweet_count = 0
        self.tweet_file = open(tweet_file_name, 'wb')
        self.tweet_writer = csv.writer(self.tweet_file, quoting=csv.QUOTE_ALL, delimiter=',')
        self.tweet_writer.writerow(['tweet'])
        super(TPStreamListener, self).__init__()

    def on_data(self, data):
        if self.tweet_count < self.tweet_limit:
            try:
                data_dict = json.loads(data)
                if (('retweeted_status' not in data_dict.keys() and 
                    data_dict['is_quote_status'] != True) and
                    len(data_dict['entities']['urls']) == 0):
                    print 'Tweet Number: ' + str(self.tweet_count)
                    tweet = str(data_dict['text'].encode('ascii','ignore'))
                    print tweet
                    self.tweet_writer.writerow([tweet])
                    print ''
                    self.tweet_count += 1
            except KeyError:
                print "No text value for tweet"
            return True
        else:
            self.tweet_file.close()
            return False

    def on_error(self, status_code):
        if status_code == 420:
            print "Hit an error"
            return False


#%% Basic version of harvesting tweets
tweet_container = []

tp_listener = TPStreamListener(tweet_limit=10000, tweet_file_name='./boyfriend_tweets_v2.csv')

tp_stream = tp.Stream(auth = api.auth, listener = tp_listener)

track_strings = [
                'need boyfriend my', 
                'want boyfriend my',
                'like boyfriend my',
                'hate boyfriend my',
                'boyfriend like to',
                'love boyfriend him',
                'like him when he boyfriend my',
                'my boyfriend is',
                'I like when my boyfriend',
                'I need boyfriend to',
                'I love boyfriend when'
                ]

tp_stream.filter(track=[','.join(track_strings)])






