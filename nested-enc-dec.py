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
    mapping[letter] = encode.onehot(random.choice(corpus))

x_train = []
roles = np.loadtxt(sys.argv[2], dtype=object)
for sentence in roles:
    x_train.append([mapping[letter] for letter in sentence])
x_train = np.array(x_train)

# Convert encodings into gestalt representations
X = [outer_encoder.predict(word) for word in x_train]
X = np.array(X)
# Reminder: Each element in X is now a python List of two numpy arrays.

# Slice X for the two numpy arrays
X_state_h = X[:,0]
X_state_c = X[:,1]
t3 = {"start": [[0,1]], "stop": [[1,0]], "none": [[0,0]]}



# Construct Inner Encoder Decoder
# ----------------------------------
# Encoder Construction
hidden_size = 100
encoder_input_t1 = keras.layers.Input(shape=(None, X.shape[3]))
encoder_input_t2 = keras.layers.Input(shape=(None, X.shape[3]))
encoder_input = keras.layers.Concatenate()([encoder_input_t1, encoder_input_t2])

encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True)
# Tie them together
encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)
# Don't need the encoder outputs, just need the states.
encoder_states = [enc_state_h, enc_state_c]

## Decoder Construction
decoder_input_t1 = keras.layers.Input(shape=(None, X.shape[3]))
decoder_input_t2 = keras.layers.Input(shape=(None, X.shape[3]))
decoder_input_t3 = keras.layers.Input(shape=(None, 2))
decoder_input = keras.layers.Concatenate()([decoder_input_t1, decoder_input_t2, decoder_input_t3])

decoder_hidden = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
# Tie it together
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=encoder_states)
decoder_dense_t1 = keras.layers.Dense(X.shape[3], activation='linear')
decoder_dense_t2 = keras.layers.Dense(X.shape[3], activation='linear')
decoder_dense_t3 = keras.layers.Dense(2, activation='linear')
# Connect output to hidden
decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]


# Finally, tie everything into a model
model = keras.Model([encoder_input_t1, encoder_input_t2, decoder_input_t1, decoder_input_t2, decoder_input_t3], decoder_output)
keras.utils.plot_model(model, to_file="new_model.png", show_shapes=True)


# Compile it...
model.compile(loss = keras.losses.MSE,
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])

# Train it
batch_size = 100
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


'''
# Get result from inner decoder
context = pass
token = np.array(encode.onehot("start"))
token = token.reshape([1, 1, token.shape[0]])
result = np.zeros([1,6,28])
output_length = 5
for x in range(output_length+1):
    out,h,c = outer_decoder.predict([token]+context)
    token = np.round(out)
    context = [h,c]
    result[0,x,:] = token
decoded_word = encode.check(result[0])
'''
