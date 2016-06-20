import tweepy as tp
import file_paths as fp
import json
import csv
import time
import unicodedata
import re

class TPStreamListener(tp.StreamListener):
    """
    Class that interacts with twitter streaming api. The key method is
    the on_data method. In that method tweets are filtered and cleaned.
    Subsequently they are written to a csv. When the tweet limit is hit
    the file is closed.
    
    """

    def __init__(self, tweet_limit=10, tweet_file_name = 'bf_tweets.csv', in_gcp=0):
        
        self.tweet_limit = tweet_limit
        self.tweet_count = 0
        self.tweet_file = open('./' + tweet_file_name, 'wb')
        if in_gcp == 1:
            gcp_tweet_file = fp.goog_file_path + tweet_file_name
            self.tweet_file = open(gcp_tweet_file, 'wb')

        self.tweet_writer = csv.writer(self.tweet_file, 
                                       quoting=csv.QUOTE_ALL, 
                                       delimiter=',')
                                       
        self.tweet_writer.writerow(['tweet'])
        super(TPStreamListener, self).__init__()

    def on_data(self, data):
        if self.tweet_count < self.tweet_limit:
            try:
                data_dict = json.loads(data)
                if (('retweeted_status' not in data_dict.keys() and 
                    data_dict['is_quote_status'] != True) and
                    len(data_dict['entities']['urls']) == 0):
                    
                    raw_tweet = str(data_dict['text'].encode('ascii','ignore'))
                    clean_tweet = clean_and_format_tweet(raw_tweet)

                    print_tweet(raw_tweet, clean_tweet, self.tweet_count)
                    
                    self.tweet_writer.writerow([clean_tweet])
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
  
   
def create_twitter_api(credentials_file_name):
    """
    Loads in credentials from a saved json file and creates an api object
    
    Inputs:
        credentials_file_name: file name of credentials file
    
    Output:
        api: api object to interact with twitter api
    """
    
    with open(credentials_file_name, 'rb') as f:
        creds = json.load(f)
        auth = tp.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
        auth.set_access_token(creds['access_token'], creds['access_token_secret'])
        api = tp.API(auth)
        return api

def print_tweet(raw_tweet, clean_tweet, tweet_count):
    """
    Prints raw and clean tweet when running script for visual inspectiion
    
    Inputs:
        raw_tweet: the tweet before cleaning
        clean_tweet: the tweet after processing
        tweet_count: which number tweet this is out of num_tweets
    
    Outputs:
        None as this just prints stuff
    """
    print "Tweet number: " + str(tweet_count)
    print "Raw form: " + raw_tweet
    print "Clean form: " + clean_tweet
    print ""

def clean_and_format_tweet(raw_tweet):
    """
    Method removes @user, #hashtags, and links from a tweet. Also gets
    rid of newline characters and makes everything lower case. This
    method definitely probably has room for improvement, but for now
    does a fairly decent job of cleaning stuff up

    To remove all punctuation and more formatting, add below to re.sub
    |([^0-9A-Za-z \t])

    Inputs:
        raw_tweet: string form of tweet that matches filters

    Outputs:
        s: The tweet after being cleaned up
    """

    s = unicodedata.normalize('NFD', unicode(raw_tweet)).encode('ascii', 'ignore')
    s = s.lower()
    s = s.replace("  ", " ")
    s = s.replace("..", ".")
    s = s.replace('\n', ' ')
    ' '.join(s.split(' '))
    s = ' '.join(re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)",
                        " ",
                        s).split())
    return s
    
def harvest_tweets(num_tweets, filters, tweet_file_name, api, in_gcp):
    """
    This is the method where the tweet listener is invoked and then
    the streaming of the twitter api begins with the filters that
    are passed

    Inputs:
        num_tweets: The number of tweets to collect
        filters: words that the tweet should contain
        tweet_file_name: name of file to save tweets into
        api: twitter api object for authentication
        in_gcp: flag for where to save file

    Outputs:
        None, this method creates a file and writes to it
        but does not actually return any output
    
    """

    tp_listener = TPStreamListener(tweet_limit= num_tweets, 
                                   tweet_file_name= tweet_file_name,
                                   in_gcp= in_gcp)
    tp_stream = tp.Stream(auth = api.auth, listener = tp_listener)
    tp_stream.filter(track=[','.join(filters)])
    
def main():
    
    api = create_twitter_api('./twitter_api_credentials.json')

    filters = [
               'need boyfriend my', 
               'want boyfriend my',
               'like boyfriend my',
               'hate boyfriend my',
               'boyfriend like to',
               'love boyfriend him',
               'please my boyfriend I',
               'refuse my boyfriend I',
               'dislike boyfriend',
               'like him when he boyfriend my',
               'my boyfriend is',
               'I like when my boyfriend',
               'I need boyfriend to',
               'I love boyfriend when'
               ]

    current_date = time.strftime('%Y_%m_%d_%I_%M_%S')
    tweet_file_name = 'boyfriend_tweets_' + str(current_date) + '.csv'
    
    harvest_tweets(100, filters, tweet_file_name, api, 0)


if __name__ == '__main__':
    main()



