import pretty_midi
import os
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import numpy as np
import pickle
import csv
from keras_preprocessing.text import Tokenizer

path_separator = os.path.sep


def get_LSTM_model(num_words):
    training_length = 50
    embedding_matrix = []
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                  input_length=training_length,
                  output_dim=100,
                  weights=[embedding_matrix],
                  trainable=False,
                  mask_zero=True))

    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(64, return_sequences=False,
                   dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))


def get_embeddings_dict():
    # embeddings_dict = {}
    # glove_path = "ass3_data" + path_separator + "glove.6B.300d.txt"
    # with open(glove_path, 'r') as f:
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         vector = np.asarray(values[1:], "float32")
    #         embeddings_dict[word] = vector

    # saving glove dictionary to pickle file for faster loading.
    # embeddings_dict = get_embeddings_dict()
    # with open('embeddings_dict.pickle', 'wb') as f:
    #     pickle.dump(embeddings_dict, f)

    # load glove dictionary from saved pickle file, for faster loading.
    embeddings_dict_path = "ass3_data" + path_separator + "embeddings_dict.pickle"
    with open(embeddings_dict_path, 'rb') as f:
        embeddings_dict = pickle.load(f)

    return embeddings_dict


def load_data_set(data_type):
    min_length = 10000
    max_length = 0
    sum_of_length = 0
    songs_artists = []
    songs_names = []
    songs_lyrics = []
    train_path = "ass3_data" + path_separator + "lyrics_" + data_type + "_set.csv"
    with open(train_path, newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in lines:
            songs_artists.append(row[0])
            songs_names.append(row[1])
            songs_lyrics.append(row[2])
            length = len(row[2])
            if length < min_length:
                min_length = length
            if length > max_length:
                max_length = length
            sum_of_length = sum_of_length + length
    print("min length song: %s" % min_length)
    print("max length song: %s" % max_length)
    print("avg length song: %s" % (sum_of_length/len(songs_lyrics)))
    return songs_artists, songs_names, songs_lyrics


def convert_words_to_integers(data):
    # prepare tokenizer
    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True,
                          split=' ')
    tokenizer.fit_on_texts(data)
    vocab_size = len(tokenizer.word_index) + 1
    # integer encode the data
    encoded_data = tokenizer.texts_to_sequences(data)
    idx_word = tokenizer.index_word
    return encoded_data, idx_word, vocab_size


def create_embedding_matrix(vocab_size, word_index, embeddings_dict):
    embedding_matrix = np.zeros((vocab_size, 300))
    for i, word in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main():

    # get embeddings_dictionary
    embeddings_dict = get_embeddings_dict()

    # load dataset
    songs_artists, songs_names, songs_lyrics = load_data_set("train")

    # tokenize the lyrics
    encoded_data, word_index, vocab_size = convert_words_to_integers(songs_lyrics)

    # create a weight matrix for words in training docs
    embedding_matrix = create_embedding_matrix(vocab_size, word_index, embeddings_dict)

    # separate encoded_data into input (X) and output (y).

    pass


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
