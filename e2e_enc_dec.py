#!/usr/bin/env python3

# @author: Blake Mullinax
# July 24th, 2020
# This is a first attempt at an end-to-end encoder-decoder
# for the sentence task. Rather than the words being fed to the
# model on their own timestep, the model will recieve all words
# at the same time, seperated by word-level start/stop tokens.

import sys

if (len(sys.argv) != 4):
    print()
    print("Usage %s <corpus> <trainingSet> <testingSet>"%(sys.argv[0]))
    print()
    sys.exit(1)
    
import keras
import pickle
import numpy as np
import string
import random
    
# Import corpus
corpus = np.loadtxt(sys.argv[1], dtype=object)

# Import roles
training_roles = np.loadtxt(sys.argv[2], dtype=object)
testing_roles = np.loadtxt(sys.argv[3], dtype=object)
    
# Create alphabet encoding
# Onehot encodings for each letter, plus start/stop tokens for
# sentences, AND start/stop tokens for words.
# 'STARTSENTENCE'/'STOPSENTENCE' - sentence level
# 'start'/'stop' - word level
onehot = keras.utils.to_categorical([i for i in range(30)])
mapping = {}
reverse_mapping = {}
for letter, encoding in zip(string.ascii_lowercase, onehot):
    mapping[letter] = encoding
mapping["STARTSENTENCE"] = onehot[26]
mapping["STOPSENTENCE"] = onehot[27]
mapping["start"] = onehot[28]
mapping["stop"] = onehot[29]

# for easy decoding later
def reverse_map(encoding):
    tokens = ["STARTSENTENCE", "STOPSENTENCE", "start", "stop"]
    i = np.argwhere(encoding == 1)[0,0]
    if i < 26:
        return string.ascii_lowercase[i]
    else:
        return tokens[i-26]
def reverse_map_all(encoding):
    word = ""
    for letter in encoding:
        word += reverse_map(letter)
    return word

# Select 10 random words from corpus and encode them
roles_to_corpus = {}
for role in string.ascii_lowercase[:10]:
    roles_to_corpus[role] = np.array([mapping[letter] for letter in random.choice(corpus)])

# Create train and testing set
x_train = []
for sentence in training_roles:
    # Start Sentence token
    encoded_sentence = mapping["STARTSENTENCE"].reshape(1,30)
    # Append each word's encoding along with word-level start/stop tokens
    for role in sentence:
        encoded_sentence = np.vstack((encoded_sentence,                       
                                      mapping["start"].reshape(1,30), 
                                      roles_to_corpus[role], 
                                      mapping["stop"].reshape(1,30)))
    # Stop Sentence token
    encoded_sentence = np.vstack((encoded_sentence, 
                                  mapping["STOPSENTENCE"].reshape(1,30)))
    x_train.append(encoded_sentence)
X = np.array(x_train)
# In this case, the target looks exactly the same as the input
Y = X.copy()

# Construct end-to-end model
hidden_size = 300
encoder_input = keras.layers.Input((None, X[0].shape[1]), name="enc_input")

encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True, name="encoder")
encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)

encoder_states = [enc_state_h, enc_state_c]

decoder_input = keras.layers.Input((None, X[0].shape[1]), name="dec_input")

decoder_hidden = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name="decoder")
decoder_hidden_output, dec_state_h, dec_state_c = decoder_hidden(decoder_input, initial_state=encoder_states)

decoder_dense = keras.layers.Dense(X[0].shape[1], activation='softmax')
decoder_output = decoder_dense(decoder_hidden_output)

model = keras.Model([encoder_input, decoder_input], decoder_output)

keras.utils.plot_model(model, to_file="./figures/test.png", show_shapes=True)

# --------
#  Train
# --------

# --------
#   Test
# --------