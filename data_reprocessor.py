# coding: utf-8
import json 
import gzip
import os
import glob
import sys
import logging
import string
import re
import datetime as dt
import numpy as np
import pandas as pd
import nltk
import NLPyPort

# declare as global so these aren't constantly loaded and unloaded from memory
NLPYPORT_CONFIG = NLPyPort.load_congif_to_list()
BR_STOPWORDS = nltk.corpus.stopwords.words("portuguese")
# regex for removing urls in text processing
PATTERN = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
TKNZR = nltk.TweetTokenizer(strip_handles=True, reduce_len=True)

logging.basicConfig(filename='reprocessor.log',  \
                filemode = 'w+',          \
                encoding='utf-8',         \
                level=logging.DEBUG)

def safeget(dct, keys):
    # AUTHOR: Diogo Pacheco
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct

def parseListOfDict(array_dict, key_to_extract, toLower=False):
    # AUTHOR: Diogo Pacheco
    if array_dict:
        result = [url.get(key_to_extract) for url in array_dict]
        if toLower:
            result = ','.join(result).lower().split(',')
        return result
    else:
        return []


def extract_hashtags(text_split):
    # function to print all the hashtags in a text
    # taken and modified from https://www.geeksforgeeks.org/python-extract-hashtags-from-text/
    # initializing hashtag_list variable
    hashtag_list = []
    # get each word from the already split text
    for word in text_split:
        # checking the first character of every word
        if word[0] == '#':
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])
    return hashtag_list

def preprocess_text(original_text):
    # Generic NLTK process taken from
    # https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline/
    urls_removed = re.sub(PATTERN, "", original_text)
    trimmed_text = urls_removed.strip()
    #cleaned_text = " ".join(trimmed_text.split())
    tokens = TKNZR.tokenize(trimmed_text)
    #punctuation_filtered = [token for token in tokens if token not in string.punctuation]
    #lowercased = [token.lower() for token in punctuation_filtered]
    #stopwords_filtered = [token for token in lowercased if token not in BR_STOPWORDS]
    #print(stopwords_filtered)
    hash_list = extract_hashtags(tokens)
    # now we can use NLPyPort for POS tagging, lemmatisation, and named entity recognition
    #print("calling NLPYPORT")
    #final_result = NLPyPort.new_full_pipe(stopwords_filtered, options = {"pre_load":True, }, config_list = NLPYPORT_CONFIG)
    #print(final_result)
    return tokens, hash_list

def addExtraEntities(entities, extra_obj_key="retweeted_status", extended_key=None):
    # AUTHOR: Diogo Pacheco
    if extended_key:
        extended_key = '.' + extended_key
    if entities.get('{}.id_str'.format(extra_obj_key)):
        extra = parseListOfDict(
            entities.get('{}{}.entities.urls'.format(extra_obj_key, extended_key), []), 
            'expanded_url'
        )
        if extra:
            entities['urls'] = entities.get('urls',[]) + extra
        extra = parseListOfDict(
            entities.get('{}{}.entities.hashtags'.format(extra_obj_key, extended_key), []), 
            'text', 
            toLower=True
        )
        if extra:
            entities['hashtags'] = entities.get('hashtags',[]) + extra
        extra = parseListOfDict(
            entities.get('{}{}.entities.user_mentions'.format(extra_obj_key, extended_key), []), 
            'id_str'
        )
        if extra:
            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + extra
        # adding quoted user to mention
        if extra_obj_key=="quoted_status":
            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + [entities.get('quoted_status.user.id_str')]

def parseTweet(tw):
    # AUTHOR: Diogo Pacheco
    entities = {
        '.'.join(k): safeget(tw,k) for k in (
            ('id_str',), 
            ("text",), # KBH: added text field because we want to use it!
            ('user','id_str'),
            ('user','screen_name'),
            ('user','followers_count'),
            ('timestamp_ms',),
            ('entities','hashtags'),
            ('entities','urls'),
            ('entities','user_mentions'),
            ('in_reply_to_status_id_str',),
            ('in_reply_to_user_id_str',),
            ('retweeted_status','id_str'),
            ('retweeted_status','user','id_str'),
            ('retweeted_status','entities','hashtags'),
            ('retweeted_status','entities','urls'),
            ('retweeted_status','entities','user_mentions'),
            ('quoted_status','id_str'),
            ('quoted_status','user','id_str'),
            ('quoted_status','entities','hashtags'),
            ('quoted_status','entities','urls'),
            ('quoted_status','entities','user_mentions'),
            ('coordinates',),
            ('place',),
            ## extended fields
            ('truncated',),
            ('extended_tweet','entities','hashtags'),
            ('extended_tweet','entities','urls'),
            ('extended_tweet','entities','user_mentions'),

            ('retweeted_status','truncated'),
            ('retweeted_status','extended_tweet','entities','hashtags'),
            ('retweeted_status','extended_tweet','entities','urls'),
            ('retweeted_status','extended_tweet','entities','user_mentions'),
            
            ('quoted_status','truncated'),
            ('quoted_status','extended_tweet','entities','hashtags'),
            ('quoted_status','extended_tweet','entities','urls'),
            ('quoted_status','extended_tweet','entities','user_mentions'),
            
#             ('place','id_str'),
#             ('place','place_type'),
#             ('place','country_code'),
#             ('place','full_name'),
#             ('place','bounding_box'),
        )
    }
#     print(entities)
    entities["user.screen_name"] = entities["user.screen_name"].lower()
    
    extended_key = 'extended_tweet' if entities.get('truncated') else ''
    
    # KBH: Modified to add text processing
    #print(entities["text"])
    processed_text, hash_list = preprocess_text(entities["text"])
    entities["text"] = processed_text
    entities["hashtags"] = hash_list
    
    if entities.get('truncated'):
        entities['urls'] = parseListOfDict(entities.get('extended_tweet.entities.urls'), 'expanded_url')
        entities['hashtags'] = parseListOfDict(entities.get('extended_tweet.entities.hashtags'), 'text', toLower=True)
        entities['user_mentions_id_str'] = parseListOfDict(entities.get('extended_tweet.entities.user_mentions'), 'id_str')
    else:
        entities['urls'] = parseListOfDict(entities.get('entities.urls'), 'expanded_url')
        entities['hashtags'] = parseListOfDict(entities.get('entities.hashtags'), 'text', toLower=True)
        entities['user_mentions_id_str'] = parseListOfDict(entities.get('entities.user_mentions'), 'id_str')
        
    if entities.get('coordinates'):
        entities['coordinates'] = entities['coordinates'].get('coordinates')
    if entities.get('place'):
        entities['place_full_name'] = entities['place'].get('full_name')
        entities['place_type'] = entities['place'].get('place_type')
        entities['place_id'] = entities['place'].get('id_str')
        entities['place_country'] = entities['place'].get('country_code')
#         bb = entities['place'].get('bounding_box',{})
#         if bb:
#             entities['place_id'] = bb.get('id_str')
#             entities['place_country'] = bb.get('country_code')
        entities.pop('place')
        
#     entities['user_mentions_name'] = parseListOfDict(entities.get('entities.user_mentions'), 'screen_name')
    
    #handling special cases of RT and quotes
    addExtraEntities(entities, extra_obj_key="retweeted_status", extended_key=extended_key)
    addExtraEntities(entities, extra_obj_key="quoted_status", extended_key=extended_key)
    
    if entities.get('retweeted_status.id_str'): # has a retweet
        extra = parseListOfDict(entities.get('retweeted_status.entities.urls',[]), 'expanded_url')
        if extra:
            entities['urls'] = entities.get('urls',[]) +  extra
        extra = parseListOfDict(entities.get('retweeted_status.entities.hashtags',[]), 'text', toLower=True)
        if extra:
            entities['hashtags'] = entities.get('hashtags',[]) + extra
        extra = parseListOfDict(entities.get('retweeted_status.entities.user_mentions',[]), 'id_str')
        if extra:
            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + extra
    
    if entities.get('quoted_status.id_str'): # has a quote
        extra = parseListOfDict(entities.get('quoted_status.entities.urls',[]), 'expanded_url')
        if extra:
            entities['urls'] = entities.get('urls',[]) +  extra
        extra = parseListOfDict(entities.get('quoted_status.entities.hashtags',[]), 'text', toLower=True)
        if extra:
            entities['hashtags'] = entities.get('hashtags',[]) + extra
        extra = parseListOfDict(entities.get('quoted_status.entities.user_mentions',[]), 'id_str')
        if extra:
            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + extra
        # adding quoted user to mention
        entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + [entities.get('quoted_status.user.id_str')]
    
    # remove duplicated urls, hashtags, and mentions to users, mainly due to nested quoted and RT.
    entities['urls'] = list(set(entities['urls']))
    entities['hashtags'] = list(set(entities['hashtags']))
    entities['user_mentions_id_str'] = list(set(entities['user_mentions_id_str']))
    
    [entities.pop(k) for k in [
        'retweeted_status.entities.urls',
        'retweeted_status.entities.hashtags',
        'retweeted_status.entities.user_mentions',
        'quoted_status.entities.urls',
        'quoted_status.entities.hashtags',
        'quoted_status.entities.user_mentions',
        'entities.urls',
        'entities.hashtags',
        'entities.user_mentions',
        
    ]]
    
    [entities.pop(k) for k,v in list(entities.items()) if ('extended' in k) or ('truncated' in k)]
    
    [entities.pop(k)for k,v in list(entities.items()) if not v]
    
    #entities['botscore'] = checkBotScoreLite(tw)
    return entities

def loadTweetsJson(filename):
    # AUTHOR: Diogo Pacheco
    init = dt.datetime.now()
    print(init, filename,)
    # with gzip.GzipFile("./data/test_elections2018_tweets-20180830.json.gz") as tw_file:
    with gzip.GzipFile(filename) as tw_file:
        data=[]
        probe_timestamp = []
        user = []
        for line in tw_file:
            try:
                tw = json.loads(line.decode('utf-8'))
                probe_timestamp.append(tw["created_at"])
                user.append(tw["user"])
                data.append(parseTweet(tw))
            except Exception as e:
                print(filename,e)
        tweets = pd.json_normalize(data)
        
        #blt_input = pd.DataFrame(probe_timestamp, columns=["probe_timestamp"])
        #blt_input["user"] = user
        #blt_scores = blt.detect_on_user_objects(blt_input.loc[:,["probe_timestamp", "user"]].values)
        #tweets["botscore"] = blt_scores["bot_score_lite"]
    
    
    tweets.drop_duplicates(subset='id_str',inplace=True)
    print(dt.datetime.now()-init, filename)
    return tweets

def main(args):
    # AUTHOR: Diogo Pacheco
    path_to_files = args[0]
#    if path_to_files.endswith('.json.gz'):
#        # convert a single file
#        tweets = loadTweetsJson(path_to_files)
#        tweets.to_parquet(path_to_files.split('.')[0]+'.parquet')
#    else:
    # convert files in a directory
    if path_to_files.endswith('/'):
        files = glob.glob(os.path.join(path_to_files, '*.json.gz'))
    #     logging.warning('\n'.join(files))
    # query string
    else:
        files = sorted(glob.glob(path_to_files))
    print("#{} files to be processed!".format(len(files)))
    
    for filename in files:
        # filename="./data/elections2018_tweets-20181003.json.gz"
        tweets = loadTweetsJson(filename)
        tweets.to_pickle(filename.split('.')[0]+'_REPROCESSED.pkl.gz')
#        tweets.to_parquet(filename.split('.')[0]+'.parquet', 
#                          #compression='gzip'
#                         )

if __name__ == '__main__':
    main(sys.argv[1:])