import os
import time

from scipy.io import arff
import numpy as np
import tensorflow as tf
from numpy import vstack
from keras.models import Sequential
from keras.layers import Dense, Dropout

path_separator = os.path.sep


def data_transformation(data, meta):
    transformed_data = []

    # for attr in meta:
    #     print(meta[attr][1])
    ctr = 0
    line_ctr = 0
    for line in data:
        transformed_line = []
        for attr in meta:
            attr_type = meta[attr][0]
            attr_range = meta[attr][1]
            if attr_type is 'numeric':
                attr_value = line[attr]
                transformed_line.append(attr_value)
                # print(line[attr])
            else:
                attr_value = str(line[attr])[2:-1]
                one_hot_vector = np.zeros(len(attr_range))
                try:
                    attr_value = attr_range.index(attr_value)
                    one_hot_vector[attr_value] = 1
                except:
                    ctr = ctr+1
                transformed_line.extend(one_hot_vector)
                # print(one_hot_vector)
        transformed_data.append(transformed_line)
        # print(line_ctr)
        line_ctr = line_ctr+1
    # print(ctr)
    return transformed_data


# define the standalone generator model
def define_generator(noise_dim, output_shape):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=noise_dim))
    model.add(Dense(output_shape, activation='linear'))
    return model


def define_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', input_dim=input_shape))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def data_batch(data, batch_size, batch_num):
    data_line = data[batch_num]
    # data_line = np.reshape(data_line, (len(data_line), 1))
    batch = data_line
    for i in range(1, batch_size):
        data_line_num = i + batch_num * batch_size
        if data_line_num < len(data):
            data_line = data[data_line_num]
            # data_line = np.reshape(data_line, (len(data_line), 1))
            batch = vstack((batch, data_line))
    return batch


# generate batch_size of real samples with class labels
def real_samples_batch(data, batch_size, batch_num):
    x = data_batch(data, batch_size, batch_num)
    # generate class labels
    y = np.ones((len(x), 1))
    return x, y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, noise_dim, n):
    noise = tf.random.normal((n, noise_dim))
    # predict outputs
    x = generator.predict(noise)
    # create class labels
    y = np.zeros((n, 1))
    return x, y


def train(data, g_model, d_model, gan_model, noise_dim, epochs, batch_size):
    # determine half the size of one batch, for updating the discriminator
    half_batch_size = int(batch_size / 2)
    # manually enumerate epochs
    for i in range(epochs):
        # prepare real samples
        x_real, y_real = real_samples_batch(data, half_batch_size, i)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, noise_dim, len(x_real))
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # generate noise as input for the generator
        noise = tf.random.normal((batch_size, noise_dim))
        x_gan = noise
        # create inverted labels for the fake samples
        y_gan = np.ones((batch_size, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)


def part1():
    # Read the data.
    data_path = 'ass4_data' + path_separator + 'adult.arff'
    data, meta = arff.loadarff(data_path)

    # apply the required data transformations.
    transformed_data = data_transformation(data, meta)

    # Define the GAN and training parameters.
    # size of the noise space
    noise_dim = 32
    output_shape = len(data[0])
    # create the discriminator
    discriminator = define_discriminator(output_shape)
    # create the generator
    generator = define_generator(noise_dim, output_shape)
    # create the gan
    gan_model = define_gan(generator, discriminator)

    # Training the GAN model.
    train(transformed_data, generator, discriminator, gan_model, noise_dim, epochs=509, batch_size=128)


def main():
    part1()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
