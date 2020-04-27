#!/usr/bin/env python3

import json
import re
import pickle

review_data = []
print("Reading data...")
with open("yelp_reviews.json", "r") as yelp:
    for object in yelp:
        review_data.append(json.loads(object))
print("Done!")
    
corpus = []
count = 0
for review in review_data:
    text = review["text"]
    # remove punctuation
    text = re.sub("[^a-zA-Z ]+", "", text).lower()
    text = text.split()
    for word in text:
        if word not in corpus:
            corpus.append(word)
            count += 1
    if count % 1000 == 0:        
        print(count)
    if count > 100500:
        break
        
print("Finished!")
print("-----------------")
# Preview Words
for i in range(100):
    print(corpus[i])
    
with open("corpus.txt", "wb") as corpus_file:
    pickle.dump(corpus, corpus_file)