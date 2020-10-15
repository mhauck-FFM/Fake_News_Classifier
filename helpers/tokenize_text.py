from nltk import word_tokenize
import pickle
import numpy as np
import string

from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

'''
FUNCTION:       tokenize
PURPOSE:        tokenize a given string and pad resulting sequence
INPUT:          full_text       (string - text that shall be tokenized)
                sequence_length (float - maximum number of words per sequence)
                tokenizer       (keras tokenizer - pretrained keras tokenizer)
OUTPUT:         tokenized text as numpy array
'''

def tokenize(full_text, sequence_length, tokenizer):

    text = full_text.translate(str.maketrans('', '', string.punctuation))

    text_tokenized = word_tokenize(text)
    text_tokenized_num = tokenizer.texts_to_sequences([text_tokenized])
    text_tokenized_num = pad_sequences(text_tokenized_num, maxlen = sequence_length, padding = 'post')

    return np.asarray(text_tokenized_num)
