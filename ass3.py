from random import randint
import pretty_midi
import os
import time
from keras import Input, Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, CuDNNLSTM, Concatenate
import numpy as np
import pickle
import csv
from numpy import savez_compressed
from numpy import load
from keras_preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import choice

path_separator = os.path.sep


def get_LSTM_model(num_words, seq_length, embedding_matrix):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                  input_length=seq_length,
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


def get_LSTM_model_2(num_words, seq_length, embedding_matrix, lyrics_input_shape=(50,),
                     melody_features_shape=(50, 108)):

    # Define the tensors for the two input
    lyrics_input = Input(lyrics_input_shape)
    melody_features_input = Input(melody_features_shape)

    # Embedding layer
    embedding = Embedding(input_dim=num_words,
                  input_length=seq_length,
                  output_dim=300,
                  weights=[embedding_matrix],
                  trainable=False,
                  mask_zero=True)(lyrics_input)

    # Masking layer for pre-trained embeddings
    masking = Masking(mask_value=0.0)(embedding)


    # add melody features to the current 300 features
    # tf.keras.layers.Concatenate(axis=0)([x, y])
    # todo: turn lyrics_input_shape from (1, 108) to (50, 108)
    # input1 need to be shape of (none, 50, 108)
    # input2 is shape of (none, 50, 300)
    # outpot need to be shape of (none, 50, 408)
    concatenate = Concatenate(axis=2)([masking, melody_features_input])

    # Recurrent layer
    lstm = LSTM(64, return_sequences=False, dropout=0.1)(concatenate)

    # Fully connected layer
    dense = Dense(64, activation='relu')(lstm)

    # Dropout for regularization
    dropout = Dropout(0.5)(dense)

    # Output layer
    prediction = Dense(num_words, activation='softmax')(dropout)

    # Connect the inputs with the outputs
    model = Model(inputs=[lyrics_input, melody_features_input], outputs=prediction)

    # return the model
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


def separate_data(encoded_data, vocab_size, seq_length):
    features = []
    labels = []
    for seq in encoded_data:
        # Create multiple training examples from each sequence
        for i in range(seq_length, len(seq)):
            # Extract the features and label
            extract = seq[i - seq_length:i + 1]

            # Set the features and label
            features.append(extract[:-1])
            labels.append(extract[-1])

    features = np.array(features)

    # make labels into one shot vectors
    one_shot_labels = np.zeros((len(features), vocab_size), dtype=np.int8)
    for i, word in enumerate(labels):
        one_shot_labels[i, word] = 1
    return features, one_shot_labels


def prepare_data(encoded_data, train_size, vocab_size, seq_length, val_data_percentage):
    # split data to train and test
    train_encoded_data, test_encoded_data = encoded_data[:train_size], encoded_data[train_size:]

    # separate encoded_data into multiple examples of input (X) and output (y).
    x_train, y_train = separate_data(train_encoded_data, vocab_size, seq_length)
    x_test, y_test = separate_data(test_encoded_data, vocab_size, seq_length)
    # label to word: word_index[np.argmax(Y[0])

    # split train to train and validation
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_data_percentage, random_state=1)
    x_train, x_val = train_val_split(train=x_train, val_data_percentage=val_data_percentage, random_seed=1)
    y_train, y_val = train_val_split(train=y_train, val_data_percentage=val_data_percentage, random_seed=1)

    return x_train, x_val, x_test, y_train, y_val, y_test


def generate_seq(model, tokenizer, seq_length, seed_text, n_words, encoded):
    result = list()
    result.append(seed_text)
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words - 1):
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


def load_midi_files(load_pickle, songs_artists, songs_names):
    all_songs_melodies = []
    pickle_file_path = "ass3_data" + path_separator + "midi_files" + ".pickle"

    if load_pickle:
        # load from saved pickle file, for faster loading.
        with open(pickle_file_path, 'rb') as f:
            all_songs_melodies = pickle.load(f)
        return all_songs_melodies

    for i in range(len(songs_artists)):
        midi_pretty_format = None
        artist = songs_artists[i]
        artist = artist.replace(" ", "_")
        song_name = songs_names[i]
        if song_name[0] is " ":
            song_name = song_name[1:]
        song_name = song_name.replace(" ", "_")
        midi_file_path = "ass3_data" + path_separator + "midi_files" + path_separator + artist + "_-_" + song_name + \
                         ".mid"
        try:
            midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_path)
        except:
            print(midi_file_path)
        all_songs_melodies.append(midi_pretty_format)

    # saving to pickle file for faster loading.
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(all_songs_melodies, f)

    return all_songs_melodies


def extract_melody_features(all_songs_melodies, total_dataset_size, seq_length, encoded_data):
    # the features for the melody are a vector of size 88 notes (0 or 1) and melody tempo total size of 89.
    # note number to name can be found at https://newt.phys.unsw.edu.au/jw/notes.html

    melody_features = np.zeros((len(all_songs_melodies), 108))  # test shape is (5, 108)
    for i in range(len(all_songs_melodies)):
        melody = all_songs_melodies[i]
        if melody is not None:
            for instrument in melody.instruments:
                for note in instrument.notes:
                    melody_features[i][note.pitch-21] = 1
            melody_features[i][107] = melody.estimate_tempo()

    melody_features_per_seq = np.empty((melody_features.shape[0], 50, 108))  # test shape is (5, 50, 108)
    for i in range(melody_features.shape[0]):
        melody_features_per_seq[i] = np.tile(melody_features[i], (50, 1))

    all_melody_features_per_seq = np.empty((total_dataset_size, 50, 108))  # test shape is (909, 50, 108)
    ctr = 0
    for i in range(len(all_songs_melodies)):
        songs_lyric = encoded_data[i]
        # melody_features_per_seq_tile = np.empty((len(songs_lyric) - seq_length, 50, 108))
        for j in range(len(songs_lyric) - seq_length):
            all_melody_features_per_seq[ctr] = melody_features_per_seq[i]
            ctr += 1
            print(ctr)

    return all_melody_features_per_seq  # test shape is (909, 50, 108)


def separate_melody_data(melody_features, songs_lyrics, seq_length):
    melody_features_seq = melody_features[0]

    songs_lyric = songs_lyrics[0]
    for j in range(1, len(songs_lyric) - seq_length):
        melody_features_seq = np.vstack((melody_features_seq, melody_features[0]))

    for i in range(1, len(songs_lyrics)):
        songs_lyric = songs_lyrics[i]
        for j in range(len(songs_lyric) - seq_length):
            melody_features_seq = np.vstack((melody_features_seq, melody_features[i]))
    return melody_features_seq


def train_val_split(train, val_data_percentage, random_seed):
    np.random.seed(random_seed)
    np.random.shuffle(train)
    val_index = int(len(train)*val_data_percentage)
    val, train = train[:val_index], train[val_index:]
    return train, val


def prepare_melody_data(train_size, val_data_percentage, all_songs_melodies, total_dataset_size, seq_length,
                        encoded_data, load_data):
    npz_train_file_path = "ass3_data" + path_separator + "melody_train_data" + ".npz"
    npz_val_file_path = "ass3_data" + path_separator + "melody_val_data" + ".npz"
    npz_test_file_path = "ass3_data" + path_separator + "melody_test_data" + ".npz"
    # load from saved file, for faster loading.
    if load_data:
        m_train = load(npz_train_file_path)['arr_0']
        m_val = load(npz_val_file_path)['arr_0']
        m_test = load(npz_test_file_path)['arr_0']
        return m_train, m_val, m_test

    # extract melody features
    melody_features = extract_melody_features(all_songs_melodies, total_dataset_size, seq_length, encoded_data)

    # split data to train and test
    m_train, m_test = melody_features[:train_size], melody_features[train_size:]

    # split train to train and validation
    m_train, m_val = train_val_split(train=m_train, val_data_percentage=val_data_percentage, random_seed=1)

    # saving to pickle file for faster loading.
    savez_compressed(npz_train_file_path, m_train)
    savez_compressed(npz_val_file_path, m_val)
    savez_compressed(npz_test_file_path, m_test)

    return m_train, m_val, m_test


def main():
    seq_length = 50

    # get embeddings_dictionary
    embeddings_dict = get_embeddings_dict(load_pickle=True)

    # load dataset
    train_songs_artists, train_songs_names, train_songs_lyrics = load_data_set(data_type="train", load_pickle=True)
    test_songs_artists, test_songs_names, test_songs_lyrics = load_data_set(data_type="test", load_pickle=True)

    # concat all lyrics
    all_songs_lyrics = []
    all_songs_lyrics.extend(train_songs_lyrics)
    all_songs_lyrics.extend(test_songs_lyrics)

    # concat all artists
    all_songs_artists = []
    all_songs_artists.extend(train_songs_artists)
    all_songs_artists.extend(test_songs_artists)

    # concat all songs names
    all_songs_names = []
    all_songs_names.extend(train_songs_names)
    all_songs_names.extend(test_songs_names)

    # load midi files melodies
    all_songs_melodies = load_midi_files(load_pickle=True,
                                         songs_artists=all_songs_artists,
                                         songs_names=all_songs_names)

    # tokenize the lyrics
    encoded_data, word_index, vocab_size, tokenizer = convert_words_to_integers(all_songs_lyrics)

    # create a weight matrix for lyrics words.
    embedding_matrix = create_embedding_matrix(vocab_size, word_index, embeddings_dict)

    # prepare data for the model.
    val_data_percentage = 0.2
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(encoded_data, len(train_songs_lyrics), vocab_size,
                                                                  seq_length, val_data_percentage)

    # prepare melody data for the model.
    total_dataset_size = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    # m_test = extract_melody_features(all_songs_melodies[600:], x_test.shape[0], seq_length, encoded_data[600:])
    train_size = total_dataset_size - x_test.shape[0]
    m_train, m_val, m_test = prepare_melody_data(train_size, val_data_percentage, all_songs_melodies,
                                                 total_dataset_size, seq_length, encoded_data, load_data=True)
    model = get_LSTM_model_2(vocab_size, seq_length, embedding_matrix)
    model.summary()

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create callbacks
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=5),
    #     ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True, save_weights_only=True)
    # ]

    # history = model.fit(x_train, y_train,
    #                     batch_size=2048, epochs=150,
    #                     callbacks=callbacks,
    #                     validation_data=(x_val, y_val))

    # model.load_weights("ass3_data" + path_separator + 'model_weights.h5')
    # score = model.evaluate(x_test, y_test, batch_size=2048)
    # print(score)

    # generate_song_lyrics(word_index, training_length, model, tokenizer)


def copy_d(m):
    m_new = np.empty((m.shape[0], 50, 108))
    for i in range(m.shape[0]):
        temp = np.tile(m[i], (50, 1))
        m_new[i] = temp
    return m_new

if __name__ == '__main__':
    start_time = time.time()
    main()
    # m_train, m_val, m_test = prepare_melody_data(0, 0, [], 0, 0, [], load_pickle=True)
    # print(m_train.shape)
    # m_new = copy_d(m_train)
    # print(m_new.shape)
    print("--- %s seconds ---" % (time.time() - start_time))
