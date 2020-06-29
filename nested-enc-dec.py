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

# Each letter that represents a role will be mapped to the encoding for a
# random word from the corpus.
mapping = {}
for letter in string.ascii_lowercase[:10]:
    onehot = encode.onehot(random.choice(corpus))
    mapping[letter] = outer_encoder.predict(np.array([onehot]))
x_train = []
roles = np.loadtxt(sys.argv[2], dtype=object)
for sentence in roles:
    x_train.append([mapping[letter] for letter in sentence])
x_train = np.array(x_train) # shape (n, 3, 2, 1, 50)
t1 = x_train[:,:,0,0,:] # new shape (n,3,50)
t2 = x_train[:,:,1,0,:] # " '' "
# 4 time steps. pre
pre_t1 = np.concatenate((np.zeros((x_train.shape[0],1,50)), t1), axis = 1)
pre_t2 = np.concatenate((np.zeros((x_train.shape[0],1,50)), t2), axis = 1)
post_t1 = np.concatenate((t1, np.zeros((x_train.shape[0],1,50))), axis = 1)
post_t2 = np.concatenate((t2, np.zeros((x_train.shape[0],1,50))), axis = 1)



# Start or stop tokens
s_s = {"start": [0,1], "stop": [1,0], "none": [0,0]}
pre_t3 = np.zeros((x_train.shape[0], 4, 2))
post_t3 = np.copy(pre_t3)
pre_t3[:,0,:] = s_s["start"]
post_t3[:,3,:] = s_s["stop"]



# Construct Inner Encoder Decoder
# ----------------------------------
# Encoder Construction
hidden_size = 100
encoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_1")
encoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_2")
encoder_input = keras.layers.Concatenate()([encoder_input_t1, encoder_input_t2])

encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True, name="encoder")
# Tie them together
encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)
# Don't need the encoder outputs, just need the states.
encoder_states = [enc_state_h, enc_state_c]

## Decoder Construction
decoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_1")
decoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_2")
decoder_input_t3 = keras.layers.Input(shape=(None, 2), name="dec_start/stop")
decoder_input = keras.layers.Concatenate()([decoder_input_t1, decoder_input_t2, decoder_input_t3])

decoder_hidden = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name="decoder")
# Tie it together
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=encoder_states)
decoder_dense_t1 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_1")
decoder_dense_t2 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_2")
decoder_dense_t3 = keras.layers.Dense(2, activation='sigmoid', name="start/stop")
# Connect output to hidden
decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]


# Finally, tie everything into a model
model = keras.Model([encoder_input_t1, encoder_input_t2, decoder_input_t1, decoder_input_t2, decoder_input_t3], decoder_output)
keras.utils.plot_model(model, to_file="new_model.png", show_shapes=True)


# Compile it...
model.compile(loss = [keras.losses.MSE, keras.losses.MSE, keras.losses.binary_crossentropy],
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])

model_input = {"enc_token_1": t1, "enc_token_2": t2, "dec_token_1": pre_t1, "dec_token_2": pre_t2, "dec_start/stop": pre_t3}
model_target = {"token_1": post_t1, "token_2": post_t2, "start/stop": post_t3}

# Train it
batch_size = 100
epochs = 800
history = model.fit(model_input, model_target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
accuracy = model.evaluate(model_input, model_target)
# use `model.metrics_names` to get indices for accuracy:
print('T1 Accuracy:', accuracy[4]*100.0, '%')
print('T2 Accuracy:', accuracy[5]*100.0, '%')
print('T3 Accuracy:', accuracy[6]*100.0, '%')



# Now remove the Teacher Forcing

# Encoder
encoder_model = keras.Model([encoder_input_t1, encoder_input_t2], encoder_states)

# Decoder
decoder_state_input_h = keras.layers.Input(shape=(hidden_size,))
decoder_state_input_c = keras.layers.Input(shape=(hidden_size,))
# inputs to hidden
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=decoder_states_input)
decoder_states = [decoder_state_h, decoder_state_c]
# hidden to outputs

decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]
decoder_model = keras.Model(
    [decoder_input_t1, decoder_input_t2, decoder_input_t3] + decoder_states_input,
    decoder_output + decoder_states)

keras.utils.plot_model(encoder_model, to_file="new_encoder.png", show_shapes=True)
keras.utils.plot_model(decoder_model, to_file="new_decoder.png", show_shapes=True)

#
## Prepare testing data
## Prepare input
#corpus = np.loadtxt(sys.argv[1], dtype=object)
#
## Each letter that represents a role will be mapped to the encoding for a
## random word from the corpus.
#x_test = []
#roles = np.loadtxt(sys.argv[3], dtype=object)
#for sentence in roles:
#    x_test.append([mapping[letter] for letter in sentence])
#x_test = np.array(x_test) # shape (n, 3, 2, 1, 50)
#t1 = x_test[:,:,0,0,:] # new shape (n,3,50)
#t2 = x_test[:,:,1,0,:] # " '' "
## 4 time steps. pre
#pre_t1 = np.concatenate((np.zeros((x_test.shape[0],1,50)), t1), axis = 1)
#pre_t2 = np.concatenate((np.zeros((x_test.shape[0],1,50)), t2), axis = 1)
#post_t1 = np.concatenate((t1, np.zeros((x_test.shape[0],1,50))), axis = 1)
#post_t2 = np.concatenate((t2, np.zeros((x_test.shape[0],1,50))), axis = 1)
#
## Start or stop tokens
#pre_t3 = np.zeros((x_test.shape[0], 4, 2))
#post_t3 = np.copy(pre_t3)
#pre_t3[:,0,:] = s_s["start"]
#post_t3[:,3,:] = s_s["stop"]
#
#
#
## Get result from inner decoder
#context = pass
#token = np.array(encode.onehot("start"))
#token = token.reshape([1, 1, token.shape[0]])
#result = np.zeros([1,6,28])
#output_length = 5
#for x in range(output_length+1):
#    out,h,c = outer_decoder.predict([token]+context)
#    token = np.round(out)
#    context = [h,c]
#    result[0,x,:] = token
#decoded_word = encode.check(result[0])
#