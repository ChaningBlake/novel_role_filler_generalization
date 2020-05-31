#!/usr/bin/env python3
import keras
import numpy as np

# Generate one-hot encoding for alphabet
# (including start and stop)
alphabet = [i for i in range(28)]
labels = keras.utils.to_categorical(alphabet, 28)
    
# Create a Dictionary of the letters to one-hot encodings
mapping = dict()
for i in range(labels.shape[0]-2):
    mapping[ chr(ord('a')+i) ] = labels[i+1]
# I know it's not necessary...but I really want the first encoding
# to be start and the last one to be stop
mapping["start"] = labels[0]
mapping["stop"] = labels[27]


def onehot(word):
    '''
    :type word: String
    :rtype: numpy array
    '''
    if word == "start" or word == "stop":
        return mapping[word]
    else:
        return [mapping[sym] for sym in word]
    
    

# A little function to check the decoder at the end
def check(result):
    for letter in result:
        for key in mapping:
            if np.array_equal(mapping[key],letter):
                print(key, end='')
    print()
    return