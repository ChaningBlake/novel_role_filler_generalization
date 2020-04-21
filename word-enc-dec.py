#!/usr/bin/env python3
# coding: utf-8
'''
___Encoder/Decoder___
November 2nd 2019
@author: Blake Mullinax

This script will construct an encoder and decoder
to test encoding different combinations of letters
into "words." It will also train and test the network.

TrainingSet and TestingSet must be preformmated like so:
abc
abb
cab

Standard Hidden Layer Size is twice the length of the input.
In this case, six.

./word-enc-dec.py trainingSet testingSet hiddenLayerSize(6) export(0 or 1)
'''


import keras
import numpy as np
import sys

if len(sys.argv) != 5:
    print()
    print("Usage: %s <trainingSet> <testingSet> <hiddenLayerSize> <export?>"%sys.argv[0])
    print()
    sys.exit(1)

# Open data sets
try:
    trainFile = open(sys.argv[1])
    testFile = open(sys.argv[2])
except:
    print("Error: Could not open files %s %s"%(sys.argv[1],sys.argv[2]))
    sys.exit(1)

length = 3
input_length = length
output_length = length

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


# A little function to check the decoder at the end
def check_decoder(result, mapping):
    for letter in result:
        for key in mapping:
            if np.array_equal(mapping[key],letter):
                print(key, end='')
    print()
    return


# encode the training set
x_train = np.loadtxt(sys.argv[1], dtype=str)
x_train = [list(i) for i in x_train]
x_train = np.array(x_train)
if x_train.shape[0] > 1:
    X=[]
    for word in x_train:
        X.append([mapping[sym] for sym in word])
    X = np.array(X)
else:
    X = np.array([mapping[i] for i in x_train])
X = X.reshape([len(x_train),length, 28])

# In this case, we want the output the same as the input
# plus start and stop encodings
Y = []
for input_vector in X:
    Y.append(np.vstack([mapping["start"], input_vector, mapping["stop"]]))
Y = np.array(Y)

# Prepare inputs for Decoder
preY = Y[:, 0:output_length+1, :]
postY = Y[:,1:output_length+2,:]


# ----Building the Net---- #


# Size of gestalt context representations
hidden_size = int(sys.argv[3])

## Encoder Construction
encoder_input = keras.layers.Input(shape=(None, X.shape[2]))
encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True)
# Tie them together
encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)
# Don't need the encoder outputs, just need the states.
encoder_states = [enc_state_h, enc_state_c]

## Decoder Construction
decoder_input = keras.layers.Input(shape=(None, preY.shape[2]))
decoder_hidden = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
# Tie it together
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=encoder_states)
decoder_dense = keras.layers.Dense(postY.shape[2], activation='softmax')
# Connect output to hidden
decoder_output = decoder_dense(decoder_hidden_output)


# Finally, tie everything into a model
model = keras.Model([encoder_input, decoder_input], decoder_output)

# Compile it...
model.compile(loss = keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])

# Train it
batch_size = 3
epochs = 400
history = model.fit([X,preY], postY,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
print('Accuracy:', model.evaluate([X,preY],postY)[1]*100.0, '%')



# Now remove the Teacher Forcing

# Encoder
encoder_model = keras.Model(encoder_input, encoder_states)

# Decoder
decoder_state_input_h = keras.layers.Input(shape=(hidden_size,))
decoder_state_input_c = keras.layers.Input(shape=(hidden_size,))
# inputs to hidden
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=decoder_states_input)
decoder_states = [decoder_state_h, decoder_state_c]
# hidden to outputs
decoder_output = decoder_dense(decoder_hidden_output)
decoder_model = keras.Model(
    [decoder_input] + decoder_states_input,
    [decoder_output] + decoder_states)



# Get the testing data ready
x_test = np.loadtxt(sys.argv[2], dtype=str)
x_test = [list(i) for i in x_test]
x_test = np.array(x_test)
if x_test.shape[0] > 1:
    test_set=[]
    for word in x_test:
        test_set.append([mapping[sym] for sym in word])
    test_set = np.array(test_set)
else:
    test_set = np.array([mapping[i] for i in x_test])
test_set = test_set.reshape([len(x_test),length, 28])



# Predict on each of the 'words'
for i in range(len(x_test)):
    # Get the context for just a single word
    context = encoder_model.predict(test_set[i:i+1])
    # Prep a sarting token
    token = np.array(mapping["start"])
    token = token.reshape([1, 1, token.shape[0]])
    
    # Get decoder's output
    result = np.zeros(postY.shape)
    for x in range(output_length+1):
        out,h,c = decoder_model.predict([token]+context)
        token = np.round(out)
        context = [h,c]
        result[:,x,:] = token
    check_decoder(result[0], mapping)
    
# Export models to JSON
if (sys.argv[4] == '1'):
    encoder_model_json = encoder_model.to_json()
    decoder_model_json = decoder_model.to_json()
    with open("encoder.json", "w") as encoder_file, open("decoder.json", "w") as decoder_file:
        encoder_file.write(encoder_model_json)
        decoder_file.write(decoder_model_json)
    encoder_model.save_weights("encoder.h5")
    decoder_model.save_weights("decoder.h5")

trainFile.close()
testFile.close()
