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
for letter, encoding in zip(string.ascii_lowercase, onehot):
    mapping[letter] = encoding
mapping["STARTSENTENCE"] = onehot[26]
mapping["STOPSENTENCE"] = onehot[27]
mapping["start"] = onehot[28]
mapping["stop"] = onehot[29]

# for easy decoding later
def reverse_map(encoding):
    tokens = ["STARTSENTENCE", "STOPSENTENCE", "start", "stop"]
    i = np.argwhere(encoding == 1)
    i = i[0,0] if i.size > 0 else None
    # Vector of zeros
    if i == None:
        return "None"
    # Alphabetic token
    elif i < 26:
        return string.ascii_lowercase[i]
    # Start stop tokens
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
Y = []
for sentence in training_roles:
    # Start Sentence token
    encoded_sentence = mapping["STARTSENTENCE"].reshape(1,30)
    # Append each word's encoding along with word-level start/stop tokens
    for role in sentence:
        encoded_sentence = np.vstack((encoded_sentence,                       
                                      mapping["start"].reshape(1,30), 
                                      roles_to_corpus[role], 
                                      mapping["stop"].reshape(1,30)))
        x_train.append(np.vstack((mapping["start"].reshape(1,30),
                                  roles_to_corpus[role],
                                  mapping["stop"].reshape(1,30))))
    # Stop Sentence token
    encoded_sentence = np.vstack((encoded_sentence, 
                                  mapping["STOPSENTENCE"].reshape(1,30)))
    Y.append(encoded_sentence)

Y = np.array(Y)
X = np.array(x_train).reshape((training_roles.shape[0], 21, 30))
preY = Y[:,:-1,:]
postY = Y[:,1:,:]

# Construct end-to-end model
hidden_size = 1024
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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# --------
#  Train
# --------
batch_size = 100
epochs = 600
history = model.fit([X,preY], postY,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0)

# Remove teacher forcing

encoder_model = keras.Model(encoder_input, encoder_states)

decoder_state_input_h = keras.layers.Input(shape=(hidden_size,))
decoder_state_input_c = keras.layers.Input(shape=(hidden_size,))

decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=decoder_states_input)
decoder_states = [decoder_state_h, decoder_state_c]
# hidden to outputs
decoder_output = decoder_dense(decoder_hidden_output)
decoder_model = keras.Model(
    [decoder_input] + decoder_states_input,
    [decoder_output] + decoder_states)

# --------
#   Test
# --------

# prepare testing data
x_test = []
Y = []
for sentence in testing_roles:
    # Start Sentence token
    encoded_sentence = mapping["STARTSENTENCE"].reshape(1,30)
    # Append each word's encoding along with word-level start/stop tokens
    for role in sentence:
        encoded_sentence = np.vstack((encoded_sentence,                       
                                      mapping["start"].reshape(1,30), 
                                      roles_to_corpus[role], 
                                      mapping["stop"].reshape(1,30)))
        x_test.append(np.vstack((mapping["start"].reshape(1,30),
                                  roles_to_corpus[role],
                                  mapping["stop"].reshape(1,30))))
    # Stop Sentence token
    encoded_sentence = np.vstack((encoded_sentence, 
                                  mapping["STOPSENTENCE"].reshape(1,30)))
    Y.append(encoded_sentence)

Y = np.array(Y)
X = np.array(x_test).reshape((testing_roles.shape[0], 21, 30))
preY = Y[:,:-1,:]
postY = Y[:,1:,:]

# Predict on each of the 'words'
word_accuracy = 0
letter_accuracy = 0
for i in range(X.shape[0]):
    # Get the context for just a single word
    context = encoder_model.predict(X[i:i+1])
    # Prep a sarting token
    token = np.array(mapping["STARTSENTENCE"])
    token = token.reshape([1, 1, token.shape[0]])
    
    # Get decoder's output
    result = np.zeros(postY.shape)
    for x in range(postY.shape[1]):
        out,h,c = decoder_model.predict([token]+context)
        token = np.round(out)
        context = [h,c]
        result[i,x,:] = token
    if np.array_equal(postY[i,:,:], result[i,:,:]):
        word_accuracy += 1
        letter_accuracy += 21
    else:
        for x in range(postY.shape[1]-1):
            if np.array_equal(postY[i,x,:], result[i,x,:]):
                letter_accuracy += 1
            else:
                pass
                #print(reverse_map(postY[i,x,:]), reverse_map(result[i,x,:]))
                
print("Word Accuracy:", word_accuracy)
print("Letter Accuracy:", letter_accuracy/(float(21)*X.shape[0])*100)