#!/usr/bin/env python3
import keras
import numpy as np
import pickle

# Open data sets
try:
    with open("short_words", "rb") as pickledData:
        corpus = pickle.load(pickledData)
except:
    print("Error: Could not open file %s"%(sys.argv[1]))
    sys.exit(1)

trainSize = 50000
length = None
input_length = 5
output_length = 5
max_length = 5    # longest word in the corpus

# Generate one-hot encoding for alphabet
# (including start and stop)
alphabet = [i for i in range(28)]
labels = keras.utils.to_categorical(alphabet, 28)

# Create a Dictionary of the letters to one-hot encodings
mapping = {}
for i in range(labels.shape[0]-2):
    mapping[ chr(ord('a')+i) ] = labels[i+1]
# I know it's not necessary...but I really want the first encoding
# to be start and the last one to be stop
mapping["start"] = labels[0]
mapping["stop"] = labels[27]


# A little function to check the decoder at the end
def check_decoder(result, mapping):
    for letter in result:
        for key in mapping:
            if np.array_equal(mapping[key],letter):
                print(key, end='')
    print()
    return

'''
# encode the training set
x_train = corpus[:trainSize]
x_train = [list(i) for i in x_train]
X = np.empty([len(x_train), max_length, 28])

for i, word in enumerate(x_train):
    temp = [mapping[sym] for sym in word]
    # fill end of word with stop tokens
    if len(temp) < max_length:
           for i in range(max_length - len(temp)):
               temp.append(mapping['stop'])
    X[i] = np.array(temp)

# In this case, we want the output the same as the input
# plus start and stop encodings
Y = []
for input_vector in X:
    Y.append(np.vstack([mapping["start"], input_vector, mapping["stop"]]))
Y = np.array(Y)

# Prepare inputs for Decoder
preY = Y[:, 0:-1, :]
postY = Y[:,1: ,:]
'''


# encode the training set
x_train = corpus[:trainSize]
x_train = [list(i) for i in x_train]
X = []
for word in x_train:
    X.append(np.array([mapping[sym] for sym in word]))
X = np.array(X)

# In this case, we want the output the same as the input
# plus start and stop encodings
Y = []
for input_vector in X:
    Y.append(np.vstack([mapping["start"], input_vector, mapping["stop"]]))
Y = np.array(Y)

# Prepare inputs for Decoder
preY = []
postY = []
for i, encoding in enumerate(Y):
    preY.append(Y[i][ 0:-1 , :])
    postY.append(Y[i][ 1: , :])
preY = np.array(preY)
postY = np.array(postY)
