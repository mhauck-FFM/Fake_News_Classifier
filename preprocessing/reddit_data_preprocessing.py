import pandas as pd
import boto3
import numpy as np
from io import StringIO
import pickle

import nltk
from nltk import word_tokenize

from gensim.models import KeyedVectors

from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, MaxPooling1D, Flatten

nltk.download('punkt')

#DEFINE MAXIMUM SEQUENCE LENGTH FOR TOKENIZED TEXT

seq_len = 25

#SPECIFY S3 RESOURCE, BUCKET AND KEY

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

bucket = 'mh-upload-bucket'
key = 'data/Reddit/'

#LOAD TRAINING, TEST, AND VALIDATION DATA FROM S3 TO PANDAS DATAFRAME

train_obj = s3_client.get_object(Bucket = bucket, Key = key + 'train.tsv')
test_obj = s3_client.get_object(Bucket = bucket, Key = key + 'test_public.tsv')
validate_obj = s3_client.get_object(Bucket = bucket, Key = key + 'validate.tsv')

df_train = pd.read_csv(train_obj['Body'], sep = '\t')
df_test = pd.read_csv(test_obj['Body'], sep = '\t')
df_validate = pd.read_csv(validate_obj['Body'], sep = '\t')

#SELECT COLUMNS 'CLEAN_TITLE' AND '6_WAY_LABEL' AS X AND Y DATA FOR THE MODEL

df_train = df_train[['clean_title', '6_way_label']]
df_test = df_test[['clean_title', '6_way_label']]
df_validate = df_validate[['clean_title', '6_way_label']]

#DROP NAN IN DATAFRAME AND CAST COLUMNS TO CORRECT DATA TYPES

df_train_finite = df_train.dropna().astype({"clean_title": str, "6_way_label": int})
df_test_finite = df_test.dropna().astype({"clean_title": str, "6_way_label": int})
df_validate_finite = df_validate.dropna().astype({"clean_title": str, "6_way_label": int})

#SEPARATE WORDS IN TITLE TO LISTS AND SAVE THEM TO NEW COLUMN IN THE RESPECTIVE DATAFRAME

df_train_finite['tokenized_title'] = df_train_finite['clean_title'].apply(word_tokenize)
df_test_finite['tokenized_title'] = df_test_finite['clean_title'].apply(word_tokenize)
df_validate_finite['tokenized_title'] = df_validate_finite['clean_title'].apply(word_tokenize)

#TRAIN TOKENIZER ON TRAINING DATA AND SAVE IT TO S3 USING PICKLE

tkz = Tokenizer(oov_token = 0, lower = True)
tkz.fit_on_texts(df_train_finite['tokenized_title'])

pickle_byte_obj = pickle.dumps(tkz)

s3_resource.Object(bucket, key + 'tokenizer_reddit.pkl').put(Body = pickle_byte_obj)

#TOKENIZE TEXTS AND PAD SEQUENCES TO THE MAXIMUM SEQUENCE LENGTH

df_train_finite['num_tokenized_title'] = df_train_finite['tokenized_title'].apply(lambda x: [x]).apply(tkz.texts_to_sequences).apply(pad_sequences, maxlen = seq_len, padding = 'post').apply(lambda x: x[0])
df_test_finite['num_tokenized_title'] = df_test_finite['tokenized_title'].apply(lambda x: [x]).apply(tkz.texts_to_sequences).apply(pad_sequences, maxlen = seq_len, padding = 'post').apply(lambda x: x[0])
df_validate_finite['num_tokenized_title'] = df_validate_finite['tokenized_title'].apply(lambda x: [x]).apply(tkz.texts_to_sequences).apply(pad_sequences, maxlen = seq_len, padding = 'post').apply(lambda x: x[0])

#SAVE FINAL DATAFRAMES TO JSON IN S3 BUCKET

json_buffer = StringIO()
df_train_finite.to_json(json_buffer, orient = 'records', lines = True)
s3_resource.Object(bucket, key + 'train_df.json').put(Body = json_buffer.getvalue())

json_buffer = StringIO()
df_test_finite.to_json(json_buffer, orient = 'records', lines = True)
s3_resource.Object(bucket, key + 'test_df.json').put(Body = json_buffer.getvalue())

json_buffer = StringIO()
df_validate_finite.to_json(json_buffer, orient = 'records', lines = True)
s3_resource.Object(bucket, key + 'validate_df.json').put(Body = json_buffer.getvalue())
