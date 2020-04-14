import time
from PIL import Image
from keras import Input, Sequential, Model
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from numpy import asarray
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import cv2

path_separator = os.path.sep


def initialize_weights(shape, dtype=None):
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initialize_bias(shape, dtype=None):
    return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype)


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # get basic model for the 2 inputs
    model = get_vgg_model(input_shape)

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def get_cnn_model(input_shape):
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (41, 41), input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (10, 10), input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7),
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    model.add(Dropout(0.01))
    return model


def get_vgg_model(input_shape):
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
    # image = Image.open(path)
    # data = asarray(image)
    data = cv2.imread(path)
    return data


def get_image_data(name, num):
    digits_to_add = 4 - len(num)
    for i in range(0, digits_to_add):
        num = '0' + num
    image_path = "dataset" + path_separator + "lfw2" + path_separator + name\
                 + path_separator + name + "_" + num + ".jpg"
    image_data = load_image(image_path)
    return image_data


def load_dataset(dataset_type):
    # consider reading all lines and shuffle them..
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
            x1_data.append(get_image_data(name=line[0], num=line[1]))
            x2_data.append(get_image_data(name=line[0], num=line[2]))
            y = 1
            y_data.append(y)
        elif len(line) == 4:
            x1_data.append(get_image_data(name=line[0], num=line[1]))
            x2_data.append(get_image_data(name=line[2], num=line[3]))
            y_data.append(y)
    file.close()
    return [x1_data, x2_data], y_data


def main():
    np.random.seed(64)

    # Local
    with open('trainShuffled3D.pickle', 'rb') as f:
        x_train, y_train = pickle.load(f)
    with open('test3D.pickle', 'rb') as f:
        x_test, y_test = pickle.load(f)

    # Colab
    # with open('/content/drive/My Drive/trainShuffled.pickle', 'rb') as f:
    #     x_train, y_train = pickle.load(f)
    # with open('/content/drive/My Drive/test.pickle', 'rb') as f:
    #     x_test, y_test = pickle.load(f)

    x1_train, x1_val, y1, y2 = train_test_split(x_train[0], y_train, test_size=0.2, random_state=1)
    x2_train, x2_val, y_train, y_val = train_test_split(x_train[1], y_train, test_size=0.2, random_state=1)
    x_train = [x1_train, x2_train]
    x_val = [x1_val, x2_val]
    model = get_siamese_model((250, 250, 3))
    model.summary()
    optimizer = Adam(lr=0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
    model.save_weights('my_model_weights.h5')
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score)

if __name__ == '__main__':
    start_time = time.time()
    main()
    # x_train, y_train = load_dataset(dataset_type="train")
    # with open('trainShuffled.pickle', 'wb') as f:
    #     pickle.dump([x_train, y_train], f)
    print("--- %s seconds ---" % (time.time() - start_time))
