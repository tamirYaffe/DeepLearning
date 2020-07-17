import time
from PIL import Image
from keras import Input, Sequential, Model
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout, \
    GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from numpy import asarray
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

path_separator = os.path.sep


def initialize_weights(shape, dtype=None):
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initialize_bias(shape, dtype=None):
    return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype)


def get_siamese_model(input_shape):
    """
        Returns the siamese model by the input shape.
        :param input_shape: the shape of the input.
        :return: a siamese model
    """
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # get basic model for the 2 inputs
    model = get_cnn_model(input_shape)

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    L1_distance = Dropout(0.4)(L1_distance)

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def get_cnn_model(input_shape):
    """
    Returns a CNN model.
    :param input_shape: the shape of the input.
    :return: a CNN model.
    """
    # Convolutional Neural Network
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(64, (41, 41), input_shape=input_shape, kernel_initializer=initialize_weights))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # 2nd Convolutional Layer
    model.add(Conv2D(64, (10, 10), kernel_initializer=initialize_weights))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # 3rd Convolutional Layer
    model.add(Conv2D(128, (7, 7), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # 4th Convolutional Layer
    model.add(Conv2D(128, (4, 4), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())

    # 5th Convolutional Layer
    model.add(Conv2D(256, (4, 4), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(
        Dense(4096, activation='sigmoid', kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    return model


def get_vgg_model(input_shape):
    """
    Returns a VGG16 based model.
    :param input_shape: the shape of the input.
    :return: a VGG16 based model.
    """
    vgg16 = VGG16(include_top=False, weights='imagenet', pooling='max', input_shape=input_shape)
    for layer in vgg16.layers[:-3]:
        layer.trainable = False
    x = vgg16.output
    x = Dense(4096, activation='sigmoid',
              kernel_regularizer=l2(1e-3),
              kernel_initializer=initialize_weights, bias_initializer=initialize_bias)(x)
    x = Dropout(0.01)(x)
    model = Model(inputs=vgg16.input, outputs=x)
    return model


def load_image(path):
    """
    Loads and return the image in the input path as binary array.
    :param path: the path of the image.
    :return: the image in the input path as binary array.
    """
    image = Image.open(path)
    data = asarray(image)
    # data = cv2.imread(path)
    return data


def get_image_data(name, num):
    """
    Loads and return the image specified by name and num as binary array.
    :param name: name of the image person face.
    :param num: number of image.
    :return: he image specified by name and num as binary array.
    """
    digits_to_add = 4 - len(num)
    for i in range(0, digits_to_add):
        num = '0' + num
    image_path = "dataset" + path_separator + "lfw2" + path_separator + name \
                 + path_separator + name + "_" + num + ".jpg"
    image_data = load_image(image_path)
    return image_data


def load_dataset(dataset_type):
    """
    Loads and return the dataset specified in the input(train/test).
    :param dataset_type: the dataset to be loaded (train/test).
    :return: the dataset specified in the input(train/test) as [x1_data, x2_data], y_data.
    """
    x1_data = []
    x2_data = []
    y_data = []
    file_path = "dataset" + path_separator + dataset_type + ".txt"
    file = open(file_path, "r")
    lines = file.readlines()
    np.random.shuffle(lines)
    for line in lines:
        line = line.split()
        y = 0
        if len(line) == 3:
            x1_data.append(np.expand_dims(get_image_data(name=line[0], num=line[1]), axis=2))
            x2_data.append(np.expand_dims(get_image_data(name=line[0], num=line[2]), axis=2))
            y = 1
            y_data.append(y)
        elif len(line) == 4:
            x1_data.append(np.expand_dims(get_image_data(name=line[0], num=line[1]), axis=2))
            x2_data.append(np.expand_dims(get_image_data(name=line[2], num=line[3]), axis=2))
            y_data.append(y)
    file.close()
    return [x1_data, x2_data], y_data


def main():
    np.random.seed(64)

    # Load local files dataset (same directory )
    # x_train, y_train = load_dataset(dataset_type="train")
    # x_test, y_test = load_dataset(dataset_type="test")

    # with pickle files
    with open('ass2_data' + path_separator + 'trainShuffled.pickle', 'rb') as f:
        x_train, y_train = pickle.load(f)
    with open('ass2_data' + path_separator + 'test.pickle', 'rb') as f:
        x_test, y_test = pickle.load(f)

    # Colab
    # with open('/content/drive/My Drive/trainShuffled.pickle', 'rb') as f:
    #     x_train, y_train = pickle.load(f)
    # with open('/content/drive/My Drive/test.pickle', 'rb') as f:
    #     x_test, y_test = pickle.load(f)

    # split train data to train and validation sets.
    x1_train, x1_val, y1, y2 = train_test_split(x_train[0], y_train, test_size=0.2, random_state=1)
    x2_train, x2_val, y_train, y_val = train_test_split(x_train[1], y_train, test_size=0.2, random_state=1)
    x_train = [x1_train, x2_train]
    x_val = [x1_val, x2_val]

    model = get_siamese_model((250, 250, 1))
    # load model
    # model = load_model('model.h5')
    # model.load_weights('my_model_weights.h5')
    # model.summary()
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
    # model.save_weights('my_model_weights.h5')

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)
    # predict_pairs(model)


def predict_pairs(model):
    """
    Predicts and prints prediction iteratively on the test set.
    :param model: the model used to predict.
    """
    x1_data = []
    x2_data = []
    file_path = "dataset" + path_separator + "test" + ".txt"
    file = open(file_path, "r")
    lines = file.readlines()
    index = 0
    best_0_line = []
    best_0_value = 1
    worst_0_line = []
    worst_0_value = 0
    best_1_line = []
    best_1_value = 0
    worst_1_line = []
    worst_1_value = 1
    for line in lines:
        line = line.split()
        y = 0
        if len(line) == 3:
            x1_data = np.expand_dims(get_image_data(name=line[0], num=line[1]), axis=2)
            x2_data = np.expand_dims(get_image_data(name=line[0], num=line[2]), axis=2)
            y = 1
        elif len(line) == 4:
            x1_data = np.expand_dims(get_image_data(name=line[0], num=line[1]), axis=2)
            x2_data = np.expand_dims(get_image_data(name=line[2], num=line[3]), axis=2)
        print(line)
        if len(line) != 1:
            x1_data = np.expand_dims(x1_data, axis=0)
            x2_data = np.expand_dims(x2_data, axis=0)
            prediction = model.predict([x1_data, x2_data])
            print("%d : class = %d, prediction = %f" % (index, y, prediction))
            if y == 0:
                if prediction < best_0_value:
                    best_0_value = prediction
                    best_0_line = line
                if prediction > worst_0_value:
                    worst_0_value = prediction
                    worst_0_line = line
            if y == 1:
                if prediction > best_1_value:
                    best_1_value = prediction
                    best_1_line = line
                if prediction < worst_1_value:
                    worst_1_value = prediction
                    worst_1_line = line
        index = index + 1
    print("best class 0 : %f, %s" % (best_0_value, best_0_line))
    print("worst class 0 : %f, %s" % (worst_0_value, worst_0_line))
    print("best class 1 : %f, %s" % (best_1_value, best_1_line))
    print("worst class 1 : %f, %s" % (worst_1_value, worst_1_line))
    file.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    # x_train, y_train = load_dataset(dataset_type="train")
    # with open('trainShuffled.pickle', 'wb') as f:
    #     pickle.dump([x_train, y_train], f)
    print("--- %s seconds ---" % (time.time() - start_time))
