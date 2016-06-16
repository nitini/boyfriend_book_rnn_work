import pandas as pd
import numpy as np
import tweepy as tp
import file_paths as fp
import json


consumer_key = 'HpbA4sFAg5ac0dXG4Sm9hflLD'
consumer_secret = 'lTYzPwYzZewGn9e0HdZgj9vtb8B9TfHvGNF7vYoUp5e2xDT31f'
access_token = '1107345421-9DUxSHtoGphHsabgWnfiUtIKmeLKy8AzCyLWVje'
access_token_secret = '1MCkGFPIB4IcM9oqhTwPnlwX5o2AXh6LA5LrcCvuAPIXr'

auth = tp.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tp.API(auth)

#%% Search using REST API



bf_tweets = api.search('"I%20want%20my%20boyfriend%20to"',lang="en")
for tweet in bf_tweets:
    print tweet.text

#%% Creating a Streaming Session

class TPStreamListener(tp.StreamListener):

    def __init__(self, tweet_limit=10, tweet_container=[]):
        self.tweet_limit = tweet_limit
        self.tweet_count = 0
        self.tweet_container = tweet_container
        super(TPStreamListener, self).__init__()

    def on_data(self, data):
        if self.tweet_count < self.tweet_limit:

            try:
                self.tweet_count += 1
                data_dict = json.loads(data)
                print 'Tweet Number: ' + str(self.tweet_count)
                print json.dumps(data_dict, indent=2, sort_keys=True)
                print data_dict['id']
                print data_dict['text']
                print ''
                self.tweet_container.append(data_dict['text'])
            except KeyError:
                print "No text value for tweet"
            return True
        else:
            return False

    def on_error(self, status_code):
        if status_code == 420:
            print "Hit an error"
            return False

#%% 743197683884560385
tweet_container = []

tp_listener = TPStreamListener(tweet_limit=2, tweet_container=tweet_container)

tp_stream = tp.Stream(auth = api.auth, listener = tp_listener)

track_strings = [
                'need boyfriend my', 
                'want boyfriend my',
                'like boyfriend my',
                'hate boyfriend my',
                ]

tp_stream.filter(track=[','.join(track_strings)])
