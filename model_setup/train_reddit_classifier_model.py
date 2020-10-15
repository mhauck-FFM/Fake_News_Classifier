import pandas as pd
import boto3
import numpy as np
import pickle

import subprocess
import sys
import os

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, Dropout, MaxPooling1D, Flatten
from keras.utils import to_categorical

import argparse

'''
FUNCTION:       install
PURPOSE:        calls pip to install a given python module
INPUT:          package       (string - desired python module)
OUTPUT:         no explicit return
'''

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('gensim')

from gensim.models import Word2Vec, KeyedVectors

#DEFINE MAIN FUNCTION FOR SAGEMAKER TRAINER

if __name__ == '__main__':

    #DEFINE PARSER AND ARGUMENTS FOR CALL WITH SAGEMAKER

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--learning_rate', type = float, default = 0.01)

    parser.add_argument('--model', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_EVAL'))

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    model_dir = args.model
    training_dir = args.train
    test_dir = args.test

    #DEFINE S3 RESOURCE, BUCKET AND KEY

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    bucket = 'mh-upload-bucket'
    key = 'data/Reddit/'

    #DEFINE MAXIMUM SEQUENCE LENGTH OF TEXTS AND MAXIMUM EMBEDDING LENGTH

    seq_len = 25
    embed_len = 300

    #LOAD TRAINING AND TEST DATA FROM S3 BUCKET AND FIT SHAPE PROPERLY

    df_train_obj = s3_client.get_object(Bucket = bucket, Key = key + 'train_df.json')
    df_test_obj = s3_client.get_object(Bucket = bucket, Key = key + 'test_df.json')

    df_train = pd.read_json(df_train_obj['Body'], orient = 'records', lines = True)
    df_test = pd.read_json(df_test_obj['Body'], orient = 'records', lines = True)

    X_train = np.concatenate(df_train['num_tokenized_title']).ravel().reshape(-1, seq_len)
    X_test = np.concatenate(df_test['num_tokenized_title']).ravel().reshape(-1, seq_len)

    #1-HOT-ENCODE CATEGORIES FOR TRAIN AND TEST DATA

    y_train = to_categorical(np.asarray(df_train['6_way_label']), num_classes = 6)
    y_test = to_categorical(np.asarray(df_test['6_way_label']), num_classes = 6)

    #LOAD TOKENIZER FROM PICKLE FILE AND SPECIFY VOCAB AND VOCAB_SIZE FOR EMBEDDING MATRIX

    tkz = pickle.loads(s3_resource.Object(bucket, key + 'tokenizer_reddit.pkl').get()['Body'].read())
    vocab_size = len(tkz.word_index) + 1
    vocab = tkz.word_index

    #LOAD PRE-TRAINED WORD2VEC MODEL (GOOGLE NEWS) AND SPECIFY EMBEDDING MATRIX

    wv = KeyedVectors.load_word2vec_format('s3://mh-upload-bucket/data/Reddit/GoogleNews-vectors-negative300.bin.gz', binary = True)
    embed_mat = np.random.normal(loc = 0., scale = 1.0, size = (vocab_size, embed_len))

    for entry, i in vocab.items():

        if entry in wv.vocab:

            embed_mat[i, :] = wv[entry]

    #DEFINE KERAS SEQUENTIAL MODEL

    model = Sequential()
    model.add(Embedding(vocab_size, embed_len, weights = [embed_mat], input_shape = (seq_len,)))
    model.add(Dropout(0.05))
    model.add(LSTM(128, dropout = 0.1, recurrent_dropout = 0.15, return_sequences = True))
    model.add(LSTM(64, dropout = 0.1, recurrent_dropout = 0.15, return_sequences = True))
    model.add(Dropout(0.10))
    model.add(LSTM(32, dropout = 0.15, recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling1D(pool_size = 2))

    model.add(Flatten())

    model.add(Dense(6, activation = 'softmax'))

    #COMPILE KERAS MODEL

    model.compile(optimizer = 'RMSprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #FIT KERAS MODEL

    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, y_test))

    #SAVE KERAS MODEL TO S3 BUCKET

    model.save('reddit_classifier.h5')
    s3_client.upload_file(Filename = 'reddit_classifier.h5', Bucket = bucket, Key = key + 'reddit_classifier.h5')
