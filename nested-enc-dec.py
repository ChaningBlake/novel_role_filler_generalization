#!/usr/bin/env python3

import keras
import sys
import numpy as np
from os import system
import encode # My module

# Import encoder and decoder model
with open('encoder_len5.json', 'r') as encoder_file, open('decoder_len5.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
encoder_model = keras.models.model_from_json(encoder_json)
decoder_model = keras.models.model_from_json(decoder_json)
encoder_model.load_weights("encoder_len5.h5")
decoder_model.load_weights("decoder_len5.h5")


# Check result
context = pass
token = np.array(encode.onehot("start"))
token = token.reshape([1, 1, token.shape[0]])
result = np.zeros([1,6,28])
output_length = 5
for x in range(output_length+1):
    out,h,c = decoder_model.predict([token]+context)
    token = np.round(out)
    context = [h,c]
    result[0,x,:] = token
decoded_word = encode.check(result[0])