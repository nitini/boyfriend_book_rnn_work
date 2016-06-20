import codecs
from keras.utils.data_utils import get_file


nietzsche_data = get_file('nietzsche.txt', 
                          origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = codecs.open(nietzsche_data, encoding='utf-8').read().lower()


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
    


#%% One time combining of first versions of twitter data
import csv
import pandas as pd
import unicodedata

with open('./boyfriend_tweets_v1.txt', 'rb') as f:
    csv_file = open('boyfriend_tweets_v1.csv','wb')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    csv_writer.writerow(['tweet'])
    for line in f:
        csv_writer.writerow([line.strip('\n')])
    csv_file.close()


bf_tweets_1 = pd.read_csv('./boyfriend_tweets_v1.csv', quoting=csv.QUOTE_ALL)
bf_tweets_2 = pd.read_csv('./boyfriend_tweets_v2_1.csv', quoting=csv.QUOTE_ALL)
bf_tweets_3 = pd.read_csv('./boyfriend_tweets_v2_2.csv', quoting=csv.QUOTE_ALL)
    
bf_tweets = pd.concat([bf_tweets_1, bf_tweets_2, bf_tweets_3], axis=0)

bf_tweets.to_csv('./old_tweet_data.csv', quoting=csv.QUOTE_ALL, index=False)

#%%


bf_tweets = pd.read_csv('./old_tweet_data.csv', quoting=csv.QUOTE_ALL)

bf_tweets['tweet'] = bf_tweets.tweet.apply(lambda x: clean_and_format_tweet(x))
bf_tweets['tweet'] = bf_tweets.tweet.apply(lambda x: x if len(x) > 30 else '')

bf_tweets = bf_tweets[bf_tweets.tweet != '' ].copy()

bf_tweets.to_csv('./old_tweet_data.csv', quoting=csv.QUOTE_ALL, index=False)
bf_tweets = pd.read_csv('./old_tweet_data.csv', quoting=csv.QUOTE_ALL)

bf_rules = pd.read_csv('./boyfriend_lines.csv', quoting=csv.QUOTE_ALL)
bf_rules['tweet'] = bf_rules.tweet.apply(lambda x: clean_and_format_tweet(x.decode('ascii','ignore')))

bf_rules = pd.concat([bf_rules]*1, ignore_index=True)

bf_tweets = pd.concat([bf_tweets, bf_rules], axis=0)

bf_tweets = bf_tweets.sample(frac=1).copy()
