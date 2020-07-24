import os
import sys
import time
from keras import Input, Model
from scipy.io import arff
import numpy as np
from numpy import vstack, hstack
from keras.models import Sequential
from keras.layers import Dense, Dropout, Concatenate, LeakyReLU, BatchNormalization, Lambda
import matplotlib.pyplot as plt
from keras import backend
from numpy.random import randn
import keras.backend as K
import tensorflow as tf
import csv

path_separator = os.path.sep
saved_models_path = "ass4_data" + path_separator + "models" + path_separator


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


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
                attr_value_category = str(line[attr])[2:-1]
                # one_hot_vector = np.zeros(len(attr_range))
                attr_value = 0
                try:
                    attr_value = attr_range.index(attr_value_category)
                    # one_hot_vector[attr_value] = 1
                except:
                    ctr = ctr + 1
                # transformed_line.extend(one_hot_vector)
                transformed_line.append(attr_value)
                # print(one_hot_vector)
        transformed_data.append(transformed_line)
        # print(line_ctr)
        line_ctr = line_ctr + 1
    print(ctr)
    return transformed_data


def data_back_transformation(transformed_data, meta):
    data = []
    for line in transformed_data:
        data_line = []
        index = 0
        for attr in meta:
            attr_type = meta[attr][0]
            attr_range = meta[attr][1]
            attr_value = int(round(line[index]))
            if attr_type is 'numeric':
                data_line.append(attr_value)
            else:
                attr_value_category = attr_range[attr_value]
                data_line.append(attr_value_category)
            index += 1
        data.append(data_line)
    return data


# define the standalone generator model
def function(X):
    # x_var = K.var(X, axis=0, keepdims=True)
    # x_var = K.sum(x_var)
    # mean, variance = tf.nn.moments(X, [0], keepdims=True)
    mean, variance = tf.nn.moments(X, [0])
    # variance = K.sum(variance, keepdims=True)
    variance = K.mean(variance, keepdims=True)
    variance = K.reshape(variance, (1, 1))
    return K.tile(variance, (K.shape(X)[0], 1))


def define_generator(noise_dim, output_shape):
    input_layer = Input(shape=(noise_dim,))
    # x = Dense(32, activation="relu")(input_layer)
    x = Dense(32)(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    # x = Dropout(0.4)(x)
    # x = Dense(64, activation="relu")(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation="relu")(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(output_shape, activation='sigmoid')(x)
    # x = Dense(output_shape-1, activation='sigmoid')(x)
    # x_var = Lambda(lambda x: function(x))(x)
    # x = Concatenate()([x, x_var])

    return Model(inputs=input_layer, outputs=x)


def define_discriminator(input_shape):
    model = Sequential()
    # model.add(Dense(128, activation="relu", kernel_initializer='he_uniform', input_dim=input_shape))
    model.add(Dense(128, kernel_initializer='he_uniform', input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    # model.add(Dense(64, activation="relu"))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    # model.add(Dense(32, activation="relu"))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(1))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.compile(loss=wasserstein_loss, optimizer='sgd')
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
    # model.compile(loss=wasserstein_loss, optimizer='adam')
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
    # mean_variance = samples_variance(x)
    # mean_variance_array = np.full((len(x), 1), mean_variance)
    # x = hstack((x, mean_variance_array))
    # generate class labels
    y = np.ones((len(x), 1))
    # y = -np.ones((len(x), 1))
    return x, y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, noise_dim, n):
    # mu, sigma = 0.5, 0.1  # mean and standard deviation
    # noise = np.random.normal(mu, sigma, (n, noise_dim))
    noise = randn(noise_dim * n)
    noise = noise.reshape(n, noise_dim)
    x = generator.predict(noise)  # noise need to be nd array
    y = np.zeros((n, 1))
    # y = np.ones((n, 1))
    return x, y


def print_progress(iterations, i, d_loss, g_loss, batch_size, total_samples):
    progress = int((i / iterations) * 20)
    bar = ""
    for j in range(20):
        if j < progress:
            bar += "="
        elif j > progress:
            bar += "."
        else:
            bar += ">"
    # print("%d/%d [%s] - d_loss: %f - g_loss: %f" % (i, iterations, bar, d_loss, g_loss))
    samples_covered = min(i * batch_size, total_samples)
    sys.stdout.write("\r%d/%d [%s] - d_loss: %f - g_loss: %f" % (samples_covered, total_samples, bar, d_loss, g_loss))


def train(data, meta, g_model, d_model, gan_model, noise_dim, epochs, batch_size, early_stop):
    # determine half the size of one batch, for updating the discriminator
    half_batch_size = int(batch_size / 2)
    iterations = int(len(data) / half_batch_size)
    if len(data) % batch_size != 0:
        iterations = iterations + 1
    history = {'d_loss': [], 'g_loss': []}
    min_joint_loss = float('inf')
    joint_loss_improvement_ctr = 0
    # manually enumerate epochs

    # worm up
    # generator = define_generator(noise_dim, meta)
    # gan = define_gan(generator, d_model)
    # for i in range(0):
    #     x_real, y_real = real_samples_batch(data, half_batch_size, i)
    #     x_fake, y_fake = generate_fake_samples(generator, noise_dim, len(x_real))
    #     d_model.train_on_batch(x_real, y_real)
    #     d_model.train_on_batch(x_fake, y_fake)
    #     noise = randn(noise_dim * batch_size)
    #     noise = noise.reshape(batch_size, noise_dim)
    #     x_gan = noise
    #     y_gan = np.ones((batch_size, 1))
    #     gan.train_on_batch(x_gan, y_gan)
    for epoch in range(epochs):
        print("Epoch (%d/%d)" % (epoch + 1, epochs))
        d_loss = 1
        g_loss = 1
        np.random.shuffle(data)
        for i in range(iterations):
            # prepare real samples
            x_real, y_real = real_samples_batch(data, half_batch_size, i)
            # prepare fake examples
            x_fake, y_fake = generate_fake_samples(g_model, noise_dim, len(x_real))
            # update discriminator
            d_real_loss = d_model.train_on_batch(x_real, y_real)
            d_fake_loss = d_model.train_on_batch(x_fake, y_fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            # d_loss = d_real_loss - d_fake_loss

            # generate noise as input for the generator
            # mu, sigma = 0.5, 0.1  # mean and standard deviation
            # noise = np.random.normal(mu, sigma, (batch_size, noise_dim))
            noise = randn(noise_dim * batch_size)
            noise = noise.reshape(batch_size, noise_dim)
            x_gan = noise
            # create inverted labels for the fake samples
            y_gan = np.ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # if (i+1) % 100 == 0:
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)
            print_progress(iterations, i, d_loss, g_loss, half_batch_size, len(data))

        # saving best model and early stop
        # joint_loss = g_loss + d_loss
        joint_loss = abs(g_loss - d_loss)
        # joint_loss = d_loss
        if min_joint_loss > joint_loss:
            joint_loss_improvement_ctr = 0
            min_joint_loss = joint_loss
            # d_model.save_weights(saved_models_path + 'discriminator_weights.h5')
            # g_model.save_weights(saved_models_path + 'generator_weights.h5')
            d_model.save(saved_models_path + 'discriminator')
            g_model.save(saved_models_path + 'generator')
        else:
            joint_loss_improvement_ctr += 1
            if joint_loss_improvement_ctr == early_stop:
                print()
                return history

        # history['d_loss'].append(d_loss)
        # history['g_loss'].append(g_loss)
        print()
    d_model.save(saved_models_path + 'discriminator')
    g_model.save(saved_models_path + 'generator')
    return history


def plot_history(history, y_scale):
    # Plot training loss values
    plt.plot(history['d_loss'])
    plt.plot(history['g_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['discriminator loss', 'generator loss'], loc='upper right')
    plt.yscale(y_scale)
    plt.show()


def find_min_dist(fake_sample, samples):
    min_dist = float('inf')
    min_dist_line = 0
    for i in range(len(samples)):
        line = samples[i]
        dist = 0
        for j in range(len(line)):
            dist += abs(line[j] - fake_sample[j])
        if dist < min_dist:
            min_dist = dist
            min_dist_line = i
    return min_dist_line, min_dist


def find_avg_dist(x_fake):
    avg_dist = 0
    for i in range(len(x_fake)):
        dist = 0
        for j in range(len(x_fake)):
            for attr in range(len(x_fake[0])):
                dist += abs(x_fake[i][attr] - x_fake[j][attr])
        dist = dist / len(x_fake)
        avg_dist += dist
    avg_dist = avg_dist / len(x_fake)
    return avg_dist


def samples_variance(samples):
    # variance = pow(samples - mean_sample, 2) / len(samples)
    mean_sample = np.mean(samples, axis=0)
    for i in range(len(samples)):
        samples[i] = pow(samples[i] - mean_sample, 2) / len(samples)
    mean_variance = np.mean(samples)
    return mean_variance


def part1(action):
    # Read the data.
    database = 'adult.arff'
    # database = 'bank-full.arff'
    data_path = 'ass4_data' + path_separator + database
    data, meta = arff.loadarff(data_path)

    # apply the required data transformations.
    transformed_data = data_transformation(data, meta)
    # x_normed = (x - x.min(0)) / x.ptp(0)
    numpy_data = np.array(transformed_data)
    min_data = numpy_data.min(0)
    max_data = numpy_data.max(0)
    normed_data = (numpy_data - min_data) / (max_data - min_data)
    # normed_data = (numpy_data - numpy_data.min(0)) / numpy_data.ptp(0)

    # Define the GAN and training parameters.
    # size of the noise space
    noise_dim = 50
    output_shape = len(normed_data[0])
    # output_shape = len(normed_data[0]) + 1
    # create the discriminator
    discriminator = define_discriminator(output_shape)
    discriminator.summary()

    # create the generator
    generator = define_generator(noise_dim, output_shape)
    generator.summary()

    # create the gan
    gan_model = define_gan(generator, discriminator)

    if action is "train":
        # Training the GAN model.
        history = train(normed_data, meta, generator, discriminator, gan_model, noise_dim,
                        epochs=30, batch_size=128, early_stop=100)
        plot_history(history, y_scale="linear")
        plot_history(history, y_scale="log")

    if action is "eval":
        # load models
        # discriminator.load_weights(saved_models_path + 'discriminator_weights.h5')
        # generator.load_weights(saved_models_path + 'generator_weights.h5')
        discriminator = tf.keras.models.load_model(saved_models_path + 'discriminator')
        generator = tf.keras.models.load_model(saved_models_path + 'generator')

        x_fake, y_fake = generate_fake_samples(generator, noise_dim, 100)
        prediction = discriminator.predict(x_fake)
        success_num = np.count_nonzero(prediction > 0.5)
        fake_samples = x_fake * (max_data - min_data) + min_data
        filter_predictions = prediction > 0.5
        filter_samples = []
        for i in range(len(prediction)):
            if filter_predictions[i]:
                filter_samples.append(fake_samples[i])
        filter_samples = np.array(filter_samples)
        success_predictions = prediction[filter_predictions]
        # max_value_index = np.argmax(prediction)
        samples_data = data_back_transformation(filter_samples, meta)
        for i in range(len(filter_samples)):
            min_dist_line, min_dist = find_min_dist(filter_samples[i], numpy_data)
            samples_data[i].append(success_predictions[i])
            samples_data[i].append(min_dist)
            samples_data[i].append(min_dist_line)
        headline = []
        for attr in meta:
            headline.append(attr)
        headline.append("prediction")
        headline.append("min dist")
        headline.append("min dist line")
        samples_data.insert(0, headline)
        with open("success_samples.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(samples_data)


def main():
    part1(action="train")
    # part1(action="eval")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
