import pandas as pd
import numpy as np
import tweepy as tp
import file_paths as fp
import json


consumer_key = 'HpbA4sFAg5ac0dXG4Sm9hflLD'
consumer_secret = 'lTYzPwYzZewGn9e0HdZgj9vtb8B9TfHvGNF7vYoUp5e2xDT31f'
access_token = '1107345421-9DUxSHtoGphHsabgWnfiUtIKmeLKy8AzCyLWVje'
access_token_secret = '1MCkGFPIB4IcM9oqhTwPnlwX5o2AXh6LA5LrcCvuAPIXr'

#%% Search using REST API

auth = tp.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tp.API(auth)

bf_tweets = api.search('"I%20want%20my%20boyfriend%20to"',lang="en")
for tweet in bf_tweets:
    print tweet.text

#%% Creating a Streaming Session

class TPStreamListener(tp.StreamListener):

    def __init__(self, tweet_limit=10):
        self.tweet_limit = tweet_limit
        self.tweet_count = 0
        super(TPStreamListener, self).__init__()

    def on_data(self, data):
        if self.tweet_count < self.tweet_limit:
            self.tweet_count += 1
            data_dict = json.loads(data)
            try:
                print "Tweet Number: " + str(self.tweet_count)
                print data_dict['text']
                print ''
            except KeyError:
                print "No text value for tweet"
            return True
        else:
            return False

    def on_error(self, status_code):
        if status_code == 420:
            print "Hit an error"
            return False

tp_listener = TPStreamListener(tweet_limit=20)
tp_stream = tp.Stream(auth = api.auth, listener = tp_listener)

track_strings = [
                'need boyfriend', 
                'want boyfriend',
                'like boyfriend',
                'hate boyfriend',
                ]

tp_stream.filter(track=[','.join(track_strings)])
