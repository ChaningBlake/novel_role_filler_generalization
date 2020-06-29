#!/usr/bin/env python3

import sys
if len(sys.argv) != 4:
    print()
    print("Usage: %s <Corpus> <TrainingSet> <TestingSet>"%(sys.argv[0]))
    print()
    sys.exit(1)
    
import keras
import numpy as np
from os import system
import encode # My module
import pickle, random, string


# Import outer decoder
with open('models/encoder_len5.json', 'r') as encoder_file, open('models/decoder_len5.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
outer_encoder = keras.models.model_from_json(encoder_json)
outer_decoder = keras.models.model_from_json(decoder_json)
outer_encoder.load_weights("models/encoder_len5.h5")
outer_decoder.load_weights("models/decoder_len5.h5")

# Prepare input
corpus = np.loadtxt(sys.argv[1], dtype=object)

mapping = {}
for letter in string.ascii_lowercase[:10]:
    onehot = encode.onehot(random.choice(corpus))
    mapping[letter] = outer_encoder.predict(np.array([onehot]))
    
x_train = []
roles = np.loadtxt(sys.argv[2], dtype=object)
for sentence in roles:
    x_train.append([mapping[letter] for letter in sentence])
x_train = np.array(x_train)
t1 = x_train[:,:,0,0,:]
t2 = x_train[:,:,1,0,:]

# Slice X for the two numpy arrays
t3 = {"start": [[0,1]], "stop": [[1,0]], "none": [[0,0]]}

X = []
for i in range(t1.shape[0]):
    X.append()
## Convert encodings into gestalt representations
#X = [outer_encoder.predict(word) for word in x_train]
#X = np.array(X)
#print(X.shape)
## Reminder: Each element in X is now a python List of two numpy arrays.
#
## Slice X for the two numpy arrays
#X_state_h = X[:,0]
#X_state_c = X[:,1]
#t3 = {"start": [[0,1]], "stop": [[1,0]], "none": [[0,0]]}
#
#print(X_state_h.shape)