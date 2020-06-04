from random import randint
import pretty_midi
import os
import time
from keras import Input, Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Concatenate, Bidirectional
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
    """
    Returns the lstm model.(version without melody features)
    :param num_words: number of words for input dim.
    :param seq_length: sequence length.
    :param embedding_matrix: matrix for the embedding layer.
    :return: the lstm model.
    """
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
                     melody_features_shape=(50, 107)):
    """
    Returns the lstm model.
    :param num_words: number of words for input dim.
    :param seq_length: sequence length.
    :param embedding_matrix: matrix for the embedding layer.
    :param lyrics_input_shape: lyrics input shape.
    :param melody_features_shape: melody features shape.
    :return: the lstm model.
    """
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
    # input1 need to be shape of (none, 50, 108)
    # input2 is shape of (none, 50, 300)
    # outpot need to be shape of (none, 50, 408)
    concatenate = Concatenate(axis=2)([masking, melody_features_input])

    # Recurrent layer
    lstm = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1))(concatenate)
    lstm = Bidirectional(LSTM(50, return_sequences=False, dropout=0.1))(lstm)

    # Fully connected layer
    # dense = TimeDistributed(Dense(1))(lstm)
    dense = Dense(128, activation='relu')(lstm)

    # Dropout for regularization
    dropout = Dropout(0.5)(dense)

    # Output layer
    prediction = Dense(num_words, activation='softmax')(dropout)

    # Connect the inputs with the outputs
    model = Model(inputs=[lyrics_input, melody_features_input], outputs=prediction)

    # return the model
    return model


def get_embeddings_dict(load_pickle):
    """
    Returnes the words embedding dictionary.
    :param load_pickle: boolean, if true then load dictionary from generated pickle file, else create dictionary
     and save to pickle file.
    :return: the words embedding dictionary.
    """
    embeddings_dict = {}
    embeddings_dict_path = "ass3_data" + path_separator + "pickle" + path_separator + "embeddings_dict.pickle"

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
    """
    Loads and returns the input data type set.
    :param data_type: train or test
    :param load_pickle: boolean, if true load the data from pickle file,
     else build the data and save it to a pickle file.
    :return: the input data type set.
    """
    songs_artists = []
    songs_names = []
    songs_lyrics = []
    pickle_file_path = "ass3_data" + path_separator + "pickle" + path_separator + data_type + ".pickle"

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
    """
    Using keras_preprocessing.text tokenizer to tokenizer the input data and returns it.
    :param data: data to tokenizer.
    :return: encoded data by the tokenizer.
    """
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
    """
    Create embedding matrix for the embedding layer to use.
    :param vocab_size: size of the vocabulary.
    :param word_index: word to index dictionary.
    :param embeddings_dict: dictionary of embedding for words.
    :return: The created embedding matrix.
    """
    embedding_matrix = np.zeros((vocab_size, 300))
    for i, word in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def separate_data(encoded_data, vocab_size, seq_length):
    """
    Separate the encoded data into sequences of input seq length and divide into features and labels.
    :param encoded_data: the encoded data.
    :param vocab_size: size of the vocabulary.
    :param seq_length: sequence length.
    :return: the Separated features and labels as one shot vectors.
    """
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
    """
    Prepare data for the training of the model.
    :param encoded_data: the encoded data.
    :param train_size: train set size.
    :param vocab_size: the vocabulary size.
    :param seq_length: sequence length.
    :param val_data_percentage: validation data set percentage.
    :return: x_train, x_val, x_test, y_train, y_val, y_test.
    """
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


def generate_seq(model, tokenizer, seed_text, n_words, encoded):
    """
    Generate song lyrics from seed text.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :param seed_text: the seed text word to start generating from.
    :param n_words: number of words to generate.
    :param encoded: encoded sequence of the seed text word.
    :return: Generated song lyrics.
    """
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


def generate_seq_with_melody_v1(model, tokenizer, seed_text, n_words, encoded, melody_features_seq):
    """
    Generate song lyrics from seed text and melody features extracted with first method.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :param seed_text: the seed text word to start generating from.
    :param n_words: number of words to generate.
    :param encoded: encoded sequence of the seed text word.
    :param melody_features_seq: melody features sequence.
    :return: Generated song lyrics.
    """
    result = list()
    result.append(seed_text)
    # generate a fixed number of words
    for _ in range(n_words):
        input_to_predict = [encoded, melody_features_seq]
        prediction = model.predict(input_to_predict)
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
        result.append(out_word)
    return ' '.join(result)


def generate_seq_with_melody_v2(model, tokenizer, seed_text, n_words, encoded, melody_features_seq):
    """
    Generate song lyrics from seed text and melody features extracted with second method.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :param seed_text: the seed text word to start generating from.
    :param n_words: number of words to generate.
    :param encoded: encoded sequence of the seed text word.
    :param melody_features_seq: melody features sequence.
    :return: Generated song lyrics.
    """
    result = list()
    result.append(seed_text)
    # generate a fixed number of words
    for i in range(n_words):
        input_to_predict = [encoded, np.expand_dims(melody_features_seq[i], axis=0)]
        prediction = model.predict(input_to_predict)
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
        result.append(out_word)
    return ' '.join(result)


def generate_song_lyrics(word_index, seq_length, model, tokenizer):
    """
    Generate song lyrics from random seed text.
    :param word_index: dictionary of words indices.
    :param seq_length: sequence length.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :return: Generated song lyrics.
    """
    # select a seed text
    # seed_text = all_songs_lyrics[randint(0, len(all_songs_lyrics))]
    seed_encode_word = randint(0, len(word_index))
    seed_text = word_index[seed_encode_word]
    encoded_test = np.zeros(seq_length)
    encoded_test[seq_length - 1] = seed_encode_word
    encoded_test = encoded_test.reshape((1, len(encoded_test)))
    print(seed_text + '\n')

    # generate new text
    generated = generate_seq(model, tokenizer, seed_text, seq_length, encoded_test)
    print(generated)


def generate_song_lyrics_with_melody_v1(word_index, seq_length, model, tokenizer, all_songs_artists, all_songs_names,
                                        all_songs_melodies, encoded_data, seed_encode_word=None,
                                        random_melody_index=None):
    """
    Generate song lyrics from input seed encoded word and melody index(choose both randomly if None).
    Using method 1 for melody features extraction.
    :param word_index: dictionary of words indices.
    :param seq_length: sequence length.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :param all_songs_artists: list of artists of all songs.
    :param all_songs_names: list of names of all songs.
    :param all_songs_melodies: list of melodies of all songs.
    :param encoded_data: the encoded data.
    :param seed_encode_word: seed encoded word to start generating lyrics from. if None then choose randomly.
    :param random_melody_index: seed melody index to base generating lyrics. if None then choose randomly.
    :return: Generated song lyrics.
    """
    # select a seed text
    # seed_text = all_songs_lyrics[randint(0, len(all_songs_lyrics))]
    if seed_encode_word is None:
        seed_encode_word = randint(0, len(word_index))
    seed_text = word_index[seed_encode_word]
    encoded_text_seq = np.zeros(seq_length)
    encoded_text_seq[seq_length - 1] = seed_encode_word
    encoded_text_seq = encoded_text_seq.reshape((1, len(encoded_text_seq)))
    print("%s : %s" % (seed_encode_word, seed_text))

    # select random melody
    if random_melody_index is None:
        random_melody_index = randint(0, len(all_songs_melodies))
    seed_song_artist = all_songs_artists[random_melody_index]
    seed_song_name = all_songs_names[random_melody_index]
    print(seed_song_artist + " - " + seed_song_name)

    # extract melody features
    seed_melody = all_songs_melodies[random_melody_index]
    songs_lyric = encoded_data[random_melody_index]
    num_of_words = len(songs_lyric)
    num_words_to_generate = num_of_words
    seed_melody_features = np.zeros(108)
    for instrument in seed_melody.instruments:
        for note in instrument.notes:
            seed_melody_features[note.pitch - 21] = 1
    seed_melody_features[107] = seed_melody.estimate_tempo()

    seed_melody_features_seq = np.empty((1, 50, 108))
    seed_melody_features_seq[0] = np.tile(seed_melody_features, (50, 1))

    # generate new text
    generated = generate_seq_with_melody_v1(model, tokenizer, seed_text, num_words_to_generate, encoded_text_seq, seed_melody_features_seq)
    print(generated+'\n')


def generate_song_lyrics_with_melody_v2(word_index, seq_length, model, tokenizer, all_songs_artists, all_songs_names,
                                        all_songs_melodies, encoded_data, seed_encode_word=None,
                                        random_melody_index=None):
    """
    Generate song lyrics from input seed encoded word and melody index(choose both randomly if None).
    Using method 2 for melody features extraction.
    :param word_index: dictionary of words indices.
    :param seq_length: sequence length.
    :param model: the LSTM model.
    :param tokenizer: the tokenizer used on the data.
    :param all_songs_artists: list of artists of all songs.
    :param all_songs_names: list of names of all songs.
    :param all_songs_melodies: list of melodies of all songs.
    :param encoded_data: the encoded data.
    :param seed_encode_word: seed encoded word to start generating lyrics from. if None then choose randomly.
    :param random_melody_index: seed melody index to base generating lyrics. if None then choose randomly.
    :return: Generated song lyrics.
    """
    # select a seed text
    # seed_text = all_songs_lyrics[randint(0, len(all_songs_lyrics))]
    if seed_encode_word is None:
        seed_encode_word = randint(0, len(word_index))
    seed_text = word_index[seed_encode_word]
    encoded_text_seq = np.zeros(seq_length)
    encoded_text_seq[seq_length - 1] = seed_encode_word
    encoded_text_seq = encoded_text_seq.reshape((1, len(encoded_text_seq)))
    print("%s : %s" % (seed_encode_word, seed_text))

    # select random melody
    if random_melody_index is None:
        random_melody_index = randint(0, len(all_songs_melodies))
    seed_song_artist = all_songs_artists[random_melody_index]
    seed_song_name = all_songs_names[random_melody_index]
    print(seed_song_artist + " - " + seed_song_name)

    # extract melody features
    seed_melody = all_songs_melodies[random_melody_index]
    songs_lyric = encoded_data[random_melody_index]
    num_of_words = len(songs_lyric)
    num_words_to_generate = num_of_words
    num_of_seq = num_of_words - seq_length
    # build word_to_notes_array
    word_to_notes = np.zeros((num_of_words, 107))
    if seed_melody is not None:
        piano_roll = seed_melody.get_piano_roll(fs=100)[21:]
        num_of_m_sec_per_word = int(piano_roll.shape[1] / num_of_words)
        for word_num in range(num_of_words):
            start = word_num * num_of_m_sec_per_word
            end = start + num_of_m_sec_per_word
            piano_roll_slice = piano_roll[:, start:end].transpose()
            piano_roll_slice_sum = np.sum(piano_roll_slice, axis=0)
            piano_roll_slice_sum[piano_roll_slice_sum > 1] = 1
            word_to_notes[word_num] = piano_roll_slice_sum

    # create seq of word_to_notes
    seed_melody_features_seq = np.empty((num_words_to_generate, seq_length, 107))

    for seq_num in range(num_words_to_generate):
        start = seq_num - seq_length + 1
        if start >= 1:
            seq = word_to_notes[start-1:start - 1 + seq_length]
            seed_melody_features_seq[seq_num] = seq
        else:
            seq = np.zeros((seq_length, 107))
            for i in range(seq_length + start):
                seq_index = i-start
                seq[seq_index] = word_to_notes[i]
            seed_melody_features_seq[seq_num] = seq

    # generate new text
    generated = generate_seq_with_melody_v2(model, tokenizer, seed_text, num_words_to_generate, encoded_text_seq, seed_melody_features_seq)
    print(generated+'\n')


def load_midi_files(load_pickle, songs_artists, songs_names):
    """
    Loads and return all songs melodies.
    :param load_pickle: if true loads from pickle saved file. else loads from data files and saves to pickle file.
    :param songs_artists: all songs artists.
    :param songs_names: all songs names.
    :return: all songs melodies.
    """
    all_songs_melodies = []
    pickle_file_path = "ass3_data" + path_separator + "pickle" + path_separator + "midi_files" + ".pickle"

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
    """
    Extract and return all songs melodies features using method 1.
    Method 1 features for the melody are a vector of size 107 for all notes appearing in all of the song(0 or 1)
     and melody tempo total size of 108.
    Note number to name can be found at https://newt.phys.unsw.edu.au/jw/notes.html.
    :param all_songs_melodies: all songs melodies.
    :param total_dataset_size: total size of the dataset of sequences.
    :param seq_length: sequence length.
    :param encoded_data: the encoded data.
    :return: all songs melodies features.
    """
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


def extract_melody_features_per_seq(all_songs_melodies, total_dataset_size, seq_length, encoded_data):
    """
    Extract and return all songs melodies features using method 2.
    Method 2 features for the melody are computed per word. they are a vector of size 107 for all notes appearing
     in the overall position of the word in the lyrics of the song(0 or 1).
    Note number to name can be found at https://newt.phys.unsw.edu.au/jw/notes.html.
    :param all_songs_melodies: all songs melodies.
    :param total_dataset_size: total size of the dataset of sequences.
    :param seq_length: sequence length.
    :param encoded_data: the encoded data.
    :return: all songs melodies features per sequence.
    """
    all_melody_features_per_seq = np.empty((total_dataset_size, 50, 107))  # test shape is (909, 50, 108)
    ctr = 0
    for i in range(len(all_songs_melodies)):
        melody = all_songs_melodies[i]
        songs_lyric = encoded_data[i]
        num_of_words = len(songs_lyric)
        num_of_seq = num_of_words - seq_length
        # build word_to_notes_array
        word_to_notes = np.zeros((num_of_words, 107))
        if melody is not None:
            piano_roll = melody.get_piano_roll(fs=100)[21:]
            num_of_m_sec_per_word = int(piano_roll.shape[1]/num_of_words)
            for word_num in range(num_of_words):
                start = word_num * num_of_m_sec_per_word
                end = start + num_of_m_sec_per_word
                piano_roll_slice = piano_roll[:, start:end].transpose()
                piano_roll_slice_sum = np.sum(piano_roll_slice, axis=0)
                piano_roll_slice_sum[piano_roll_slice_sum > 1] = 1
                word_to_notes[word_num] = piano_roll_slice_sum

        # create seq of word_to_notes
        for seq_num in range(num_of_seq):
            seq = word_to_notes[seq_num:seq_num + seq_length]
            all_melody_features_per_seq[ctr] = seq
            ctr += 1
            print(ctr)

    return all_melody_features_per_seq  # test shape is (909, 50, 108)


def separate_melody_data(melody_features, songs_lyrics, seq_length):
    """
    Separate and return melody features for the song lyrics.
    :param melody_features: the melody features.
    :param songs_lyrics: songs lyrics.
    :param seq_length: the sequence length.
    :return: Melody features for the song lyrics.
    """
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
    """
    Split train data into train and validation sets by the input validation percentage.
    :param train: the training data.
    :param val_data_percentage: validation percentage.
    :param random_seed: random to seed to use, in order to replicate results.
    :return: train and validation sets.
    """
    np.random.seed(random_seed)
    np.random.shuffle(train)
    val_index = int(len(train)*val_data_percentage)
    val, train = train[:val_index], train[val_index:]
    return train, val


def prepare_melody_data(train_size, val_data_percentage, all_songs_melodies, total_dataset_size, seq_length,
                        encoded_data, load_data):
    """
    Prepare melody data for the model training.
    :param train_size: the training size.
    :param val_data_percentage: validation data percentage.
    :param all_songs_melodies: all songs melodies.
    :param total_dataset_size: total dataset size of sequences.
    :param seq_length: sequence length.
    :param encoded_data: the encoded data.
    :param load_data: if true loads data from npz files, else prepare the data and save to npz files.
    :return: melody train, melody validation, and melody test data.
    """
    npz_train_file_path = "ass3_data" + path_separator + "npz" + path_separator + "melody_train_data" + ".npz"
    npz_val_file_path = "ass3_data" + path_separator + "npz" + path_separator + "melody_val_data" + ".npz"
    npz_test_file_path = "ass3_data" + path_separator + "npz" + path_separator + "melody_test_data" + ".npz"
    # load from saved file, for faster loading.
    if load_data:
        m_train = load(npz_train_file_path)['arr_0']
        m_val = load(npz_val_file_path)['arr_0']
        m_test = load(npz_test_file_path)['arr_0']
        return m_train, m_val, m_test

    # extract melody features
    melody_features = extract_melody_features_per_seq(all_songs_melodies, total_dataset_size, seq_length, encoded_data)

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
    # m_test = extract_melody_features_per_seq(all_songs_melodies[600:], x_test.shape[0], seq_length, encoded_data[600:])
    train_size = total_dataset_size - x_test.shape[0]
    m_train, m_val, m_test = prepare_melody_data(train_size, val_data_percentage, all_songs_melodies,
                                                 total_dataset_size, seq_length, encoded_data, load_data=True)

    # add melody to train, val and test data
    x_train = [x_train, m_train]
    x_val = [x_val, m_val]
    x_test = [x_test, m_test]

    model = get_LSTM_model_2(vocab_size, seq_length, embedding_matrix)
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

    model.load_weights("ass3_data" + path_separator + 'model_weights' + path_separator + 'v2' + path_separator + 'model_weights.h5')
    # score = model.evaluate(x_test, y_test, batch_size=2048)
    # print(score)

    # np.random.seed(1)
    seed_encode_words = [5640, 2088, 4764]
    for i in range(3):
        #  choose random word
        # seed_encode_word = randint(0, len(word_index))
        seed_encode_word = seed_encode_words[i]
        for j in range(5):
            #  generate lyrics for each of the test melodies

            # generate_song_lyrics_with_melody_v1(word_index, seq_length, model, tokenizer,
            #                                     all_songs_artists,
            #                                     all_songs_names,
            #                                     all_songs_melodies,
            #                                     encoded_data,
            #                                     seed_encode_word=seed_encode_word,
            #                                     random_melody_index=600+j)


            generate_song_lyrics_with_melody_v2(word_index, seq_length, model, tokenizer,
                                                all_songs_artists,
                                                all_songs_names,
                                                all_songs_melodies,
                                                encoded_data,
                                                seed_encode_word=seed_encode_word,
                                                random_melody_index=600+j)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
