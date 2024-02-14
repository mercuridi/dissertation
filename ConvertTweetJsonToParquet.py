# coding: utf-8
import json 
import gzip
import os
import glob
import sys
import logging
import datetime as dt
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

#from BotometerLite.BotometerLite.core import BotometerLiteDetector

#blt = BotometerLiteDetector()

#def checkBotScoreLite(tw_json):
    #btl = BotometerLiteDetector()
    #return btl.detect_on_tweet_objects([tw_json]).loc[0,"bot_score_lite"]

# In[2]:


def safeget(dct, keys):
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


# In[3]:


def parseListOfDict(array_dict, key_to_extract, toLower=False):
    if array_dict:
        result = [url.get(key_to_extract) for url in array_dict]
        if toLower:
            result = ','.join(result).lower().split(',')
        return result
    else:
        return []

def addExtraEntities(entities, extra_obj_key="retweeted_status", extended_key=None):
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
    entities = {
        '.'.join(k): safeget(tw,k) for k in (
            ('id_str',), 
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
    
#    if entities.get('retweeted_status.id_str'): # has a retweet
#        extra = parseListOfDict(entities.get('retweeted_status.entities.urls',[]), 'expanded_url')
#        if extra:
#            entities['urls'] = entities.get('urls',[]) +  extra
#        extra = parseListOfDict(entities.get('retweeted_status.entities.hashtags',[]), 'text', toLower=True)
#        if extra:
#            entities['hashtags'] = entities.get('hashtags',[]) + extra
#        extra = parseListOfDict(entities.get('retweeted_status.entities.user_mentions',[]), 'id_str')
#        if extra:
#            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + extra
    
#    if entities.get('quoted_status.id_str'): # has a quote
#        extra = parseListOfDict(entities.get('quoted_status.entities.urls',[]), 'expanded_url')
#        if extra:
#            entities['urls'] = entities.get('urls',[]) +  extra
#        extra = parseListOfDict(entities.get('quoted_status.entities.hashtags',[]), 'text', toLower=True)
#        if extra:
#            entities['hashtags'] = entities.get('hashtags',[]) + extra
#        extra = parseListOfDict(entities.get('quoted_status.entities.user_mentions',[]), 'id_str')
#        if extra:
#            entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + extra
#        # adding quoted user to mention
#        entities['user_mentions_id_str'] = entities.get('user_mentions_id_str',[]) + [entities.get('quoted_status.user.id_str')]
    
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


# In[41]:


def loadTweetsJson(filename):
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
        tweets = json_normalize(data)
        
        #blt_input = pd.DataFrame(probe_timestamp, columns=["probe_timestamp"])
        #blt_input["user"] = user
        #blt_scores = blt.detect_on_user_objects(blt_input.loc[:,["probe_timestamp", "user"]].values)
        #tweets["botscore"] = blt_scores["bot_score_lite"]
    
    
    tweets.drop_duplicates(subset='id_str',inplace=True)
    print(dt.datetime.now()-init, filename)
    return tweets


# In[42]:

def main(args):
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
        tweets.to_pickle(filename.split('.')[0]+'.pkl.gz')
#        tweets.to_parquet(filename.split('.')[0]+'.parquet', 
#                          #compression='gzip'
#                         )

if __name__ == '__main__':
    main(sys.argv[1:])