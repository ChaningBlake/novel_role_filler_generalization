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


# Import inner decoder
with open('encoder_len5.json', 'r') as encoder_file, open('decoder_len5.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
inner_encoder = keras.models.model_from_json(encoder_json)
inner_decoder = keras.models.model_from_json(decoder_json)
inner_encoder.load_weights("encoder_len5.h5")
inner_decoder.load_weights("decoder_len5.h5")

# Prepare input
corpus = np.loadtxt(sys.argv[1], dtype=object)

mapping = {}
for letter in string.ascii_lowercase[:10]:
   mapping[letter] = encode.onehot(random.choice(corpus))

# Match letter combinations with words e.g. cef -> John Ate Fish
x_train = []
roles = np.loadtxt(sys.argv[2], dtype=object)
for sentence in roles:
    x_train.append([mapping[letter] for letter in sentence])
x_train = np.array(x_train)

# Fill X with the gestalt representations for each word.
X = []
print(inner_encoder.predict(x_train[0:1,0,:]))

'''
for i in range(len(x_train)):
    sentence = []
    for j in range(3):
        sentence.append(inner_encoder.predict(x_train[i:i+1,j,:]))
    X.append(sentence)
X = np.array(X)
'''









'''
#!/usr/bin/env python3
import keras
import numpy as np
import pickle
import encode

hidden_size = 50


X = []
for i in range(100):
    X.append(np.round(np.random.rand(3,50)))
X = np.array(X)
start = np.zeros(50)
start[0] = 1
stop = np.zeros(50)
stop[49] = 1

Y = []
for input_vector in X:
    Y.append(np.vstack([start, input_vector, stop]))
Y = np.array(Y)

preY = Y[:, 0:-1, :]
postY = Y[:, 1:, :]

# Encoder Construction
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