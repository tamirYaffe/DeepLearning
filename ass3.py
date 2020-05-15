import pretty_midi
import os
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import numpy as np
import pickle

path_separator = os.path.sep
num_words = 300


def get_LSTM_model():
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
    embeddings_dict = {}
    glove_path = "ass3_data" + path_separator + "glove.6B.300d.txt"
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def main():
    # pm = pretty_midi.PrettyMIDI('2_Unlimited_-_Get_Ready_for_This.mid')

    # saving glove dictionary to pickle file for faster loading.
    # embeddings_dict = get_embeddings_dict()
    # with open('embeddings_dict.pickle', 'wb') as f:
    #     pickle.dump(embeddings_dict, f)

    # load glove dictionary from saved pickle file, for faster loading.
    embeddings_dict_path = "ass3_data" + path_separator + "embeddings_dict.pickle"
    with open(embeddings_dict_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    pass



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))