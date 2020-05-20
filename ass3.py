from random import randint
import pretty_midi
import os
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, CuDNNLSTM
import numpy as np
import pickle
import csv

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import choice
path_separator = os.path.sep


def get_LSTM_model(num_words, training_length, embedding_matrix):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                  input_length=training_length,
                  output_dim=300,
                  weights=[embedding_matrix],
                  trainable=False,
                  mask_zero=True))

    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))  # output shape is [none, 50, 300]

    # add melody features to the current 300 features
    # tf.keras.layers.Concatenate(axis=0)([x, y])

    # Recurrent layer
    model.add(LSTM(64, return_sequences=False,
                   dropout=0.1, recurrent_dropout=0.1))
    # model.add(CuDNNLSTM(64, return_sequences=False))
    # model.add(Dropout(0.1))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    return model


def get_embeddings_dict(load_pickle):
    embeddings_dict = {}
    embeddings_dict_path = "ass3_data" + path_separator + "embeddings_dict.pickle"

    if load_pickle:
        # load from saved pickle file, for faster loading.
        with open(embeddings_dict_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        return embeddings_dict

    glove_path = "ass3_data" + path_separator + "glove.6B.300d.txt"
    with open(glove_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # saving to pickle file for faster loading.
    with open(embeddings_dict_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict


def load_data_set(data_type, load_pickle):
    songs_artists = []
    songs_names = []
    songs_lyrics = []
    pickle_file_path = "ass3_data" + path_separator + data_type + ".pickle"

    if load_pickle:
        # load from saved pickle file, for faster loading.
        with open(pickle_file_path, 'rb') as f:
            songs_artists, songs_names, songs_lyrics = pickle.load(f)
        return songs_artists, songs_names, songs_lyrics

    min_length = 10000
    max_length = 0
    sum_of_length = 0
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
    print("avg length song: %s" % (sum_of_length / len(songs_lyrics)))

    # saving to pickle file for faster loading.
    with open(pickle_file_path, 'wb') as f:
        pickle.dump([songs_artists, songs_names, songs_lyrics], f)

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
    return encoded_data, idx_word, vocab_size, tokenizer


def create_embedding_matrix(vocab_size, word_index, embeddings_dict):
    embedding_matrix = np.zeros((vocab_size, 300))
    for i, word in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def separate_data(encoded_data, vocab_size, training_length):
    features = []
    labels = []

    for seq in encoded_data:

        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length:i + 1]

            # Set the features and label
            features.append(extract[:-1])
            labels.append(extract[-1])

    features = np.array(features)

    # make labels into one shot vectors
    one_shot_labels = np.zeros((len(features), vocab_size), dtype=np.int8)
    for i, word in enumerate(labels):
        one_shot_labels[i, word] = 1
    return features, one_shot_labels


def prepare_data(encoded_data, train_size, vocab_size, training_length):
    # split data to train and test
    train_encoded_data, test_encoded_data = encoded_data[:train_size], encoded_data[train_size:]

    # separate encoded_data into multiple examples of input (X) and output (y).
    x_train, y_train = separate_data(train_encoded_data, vocab_size, training_length)
    x_test, y_test = separate_data(test_encoded_data, vocab_size, training_length)
    # label to word: word_index[np.argmax(Y[0])

    # split train to train and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def generate_seq(model, tokenizer, seq_length, seed_text, n_words, encoded):
    result = list()
    result.append(seed_text)
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words-1):
        # encode the text as integer
        # encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        # encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # encoded_test = np.zeros(seq_length)
        # encoded_test[seq_length-1] = encoded_word
        # encoded_test = encoded_test.reshape((1, len(encoded_test)))
        # predict probabilities for each word
        # yhat = model.predict_classes(encoded, verbose=0)
        prediction = model.predict(encoded)
        prediction = prediction.reshape(prediction.size)
        indecies = np.arange(prediction.size)
        draw = choice(indecies, 1, p=prediction)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == draw:
                out_word = word
                break
        # append to input
        encoded.reshape(encoded.size)
        encoded = np.c_[encoded, draw]
        encoded = np.delete(encoded, 1, 1)
        encoded.reshape((1, encoded.size))
        # in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


def generate_song_lyrics(word_index, training_length, model, tokenizer):
    # select a seed text
    # seed_text = all_songs_lyrics[randint(0, len(all_songs_lyrics))]
    seed_encode_word = randint(0, len(word_index))
    seed_text = word_index[seed_encode_word]
    encoded_test = np.zeros(training_length)
    encoded_test[training_length - 1] = seed_encode_word
    encoded_test = encoded_test.reshape((1, len(encoded_test)))
    print(seed_text + '\n')

    # generate new text
    generated = generate_seq(model, tokenizer, training_length, seed_text, training_length, encoded_test)
    print(generated)


def main():
    training_length = 50

    # get embeddings_dictionary
    embeddings_dict = get_embeddings_dict(load_pickle=True)

    # load dataset
    train_songs_artists, train_songs_names, train_songs_lyrics = load_data_set(data_type="train", load_pickle=True)
    test_songs_artists, test_songs_names, test_songs_lyrics = load_data_set(data_type="test", load_pickle=True)

    # concat all lyrics
    all_songs_lyrics = []
    all_songs_lyrics.extend(train_songs_lyrics)
    all_songs_lyrics.extend(test_songs_lyrics)

    # tokenize the lyrics
    encoded_data, word_index, vocab_size, tokenizer = convert_words_to_integers(all_songs_lyrics)

    # create a weight matrix for lyrics words.
    embedding_matrix = create_embedding_matrix(vocab_size, word_index, embeddings_dict)

    # prepare data for the model.
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(encoded_data, len(train_songs_lyrics), vocab_size,
                                                                  training_length)

    model = get_LSTM_model(vocab_size, training_length, embedding_matrix)
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True, save_weights_only=True)
    ]

    # history = model.fit(x_train, y_train,
    #                     batch_size=2048, epochs=150,
    #                     callbacks=callbacks,
    #                     validation_data=(x_val, y_val))

    model.load_weights("ass3_data" + path_separator + 'model_weights.h5')
    # score = model.evaluate(x_test, y_test, batch_size=2048)
    # print(score)

    generate_song_lyrics(word_index, training_length, model, tokenizer)



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
