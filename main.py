import json
import collections
import time
import pandas as pd
import spacy
import pt_core_news_sm

start_time = time.time()
print("Processing pickle...")
obj = pd.read_pickle(r'elections2018_tweets-20180830.pkl')
bots = obj.loc[obj['botscore'] > 0.7]
bot_id_strs = set(bots["id_str"])
print(f"Pickle processed. ID set size: {len(bot_id_strs)}")
#print(bots)
#print(bot_id_strs)

print("Processing raw JSON data...")
raw_data = []
with open('elections2018_tweets-20180830.json', encoding='utf8') as json_file:
    for line in json_file:
        line_data = json.loads(line)
        if line_data["id_str"] in bot_id_strs:
            raw_data.append(line_data["text"])
print("JSON processed.")

print("NLP start...")
nlp_start = time.time()
print("Loading Portuguese NLP library.")
nlp = spacy.load("pt_core_news_sm")
nlp = pt_core_news_sm.load()
print("Portuguese NLP loaded.")

print("Performing NLP via spaCy2...")
nlp_processed = []
for text in raw_data:
    doc = nlp(text)
    nlp_processed.append(doc)
    #print([(w.text, w.pos_) for w in doc])
end_time = time.time()
print(f"NLP processing complete: {len(nlp_processed)} records.")

time_elapsed = end_time - start_time
nlp_time = end_time - nlp_start
print(f"Time elapsed: {time_elapsed}")
print(f"Of which NLP: {nlp_time}")
