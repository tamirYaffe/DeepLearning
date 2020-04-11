import time
from PIL import Image
from keras import Input, Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.regularizers import l2
from keras import backend as K
from numpy import asarray
import os

path_separator = os.path.sep


def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return K.random_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return K.random_normal(shape, mean=0.5, stddev=0.01, dtype=dtype)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


def load_image(path):
    image = Image.open(path)
    data = asarray(image)
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
    dataset = []
    file_path = "dataset" + path_separator + dataset_type + ".txt"
    file = open(file_path, "r")
    file.readline()
    for line in file:
        line = line.split()
        x = []
        y = 0
        if len(line) == 3:
            x.append(get_image_data(name=line[0], num=line[1]))
            x.append(get_image_data(name=line[0], num=line[2]))
            y = 1
        elif len(line) == 4:
            x.append(get_image_data(name=line[0], num=line[1]))
            x.append(get_image_data(name=line[2], num=line[3]))
        data_line = {'X': x, 'Y': y}
        dataset.append(data_line)
    file.close()
    return dataset


def main():
    # train = load_dataset(dataset_type="train")
    # test = load_dataset(dataset_type="test")
    model = get_siamese_model((105, 105, 1))
    model.summary()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
