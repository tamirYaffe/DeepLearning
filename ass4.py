import os
import pickle
import sys
import time
from keras import Input, Model
from keras.optimizers import Adam
from scipy.io import arff
import numpy as np
from numpy import vstack
from keras.models import Sequential
from keras.layers import Dense, Dropout, Concatenate, LeakyReLU, BatchNormalization, Lambda, Conv2D, Flatten, \
    Conv2DTranspose, Reshape
import matplotlib.pyplot as plt
from keras import backend
from numpy.random import randn
import keras.backend as K
import tensorflow as tf
import csv
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path_separator = os.path.sep
saved_models_path = "ass4_data" + path_separator + "models" + path_separator
# random_forest = RandomForestClassifier(n_estimators=100)
random_forest_filename = 'random_forest_model'
test_file_name = 'train_test'


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
    # print(ctr)
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


def define_deep_generator(noise_dim, output_shape):
    model = Sequential()

    model.add(Conv2DTranspose(filters=32, kernel_size=(2, 2), input_shape=noise_dim))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=16, kernel_size=(4, 4)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(filters=8, kernel_size=(8, 8)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='sigmoid'))
    return model


def define_deep_discriminator(input_shape):
    model = Sequential()

    model.add(Dense(3528, input_shape=(input_shape,)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((21, 21, 8)))

    model.add(Conv2D(filters=8, kernel_size=(8, 8)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=16, kernel_size=(4, 4)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=32, kernel_size=(2, 2)))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


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
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.compile(loss='binary_crossentropy', optimizer='sgd')
    # model.compile(loss=wasserstein_loss, optimizer='sgd')
    return model


def define_discriminator2(input_shape):
    input_layer = Input(shape=(input_shape,))

    input_layer_var = Lambda(lambda x: function(x))(input_layer)
    x = Concatenate()([input_layer, input_layer_var])

    # x = Dense(32, activation="relu")(input_layer)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    # x = Dense(64, activation="relu")(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    # x = Dense(128, activation="relu")(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(1, activation='sigmoid')(x)
    # x = Dense(output_shape-1, activation='sigmoid')(x)
    # x_var = Lambda(lambda x: function(x))(x)
    # x = Concatenate()([x, x_var])

    model = Model(inputs=input_layer, outputs=x)
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer='adam')
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
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
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
    # y = np.ones((len(x), 1))
    y = np.random.uniform(0.7, 1.2, (len(x), 1))
    # y = np.full((len(x), 1), 0.9)
    # y = -np.ones((len(x), 1))
    return x, y


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, noise_dim, n):
    # mu, sigma = 0.5, 0.1  # mean and standard deviation
    # noise = np.random.normal(mu, sigma, (n, noise_dim))
    # noise = randn(noise_dim * n)
    # noise = noise.reshape(n, noise_dim)

    noise = randn(noise_dim[0] * noise_dim[1] * n)
    noise = noise.reshape((n, noise_dim[0], noise_dim[1], noise_dim[2]))

    x = generator.predict(noise)  # noise need to be nd array
    # y = np.zeros((n, 1))
    y = np.random.uniform(0, 0.3, (len(x), 1))
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
            # noise = randn(noise_dim * batch_size)
            # noise = noise.reshape(batch_size, noise_dim)
            noise = randn(noise_dim[0] * noise_dim[1] * batch_size)
            noise = noise.reshape((batch_size, noise_dim[0], noise_dim[1], noise_dim[2]))
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
        # joint_loss = abs(g_loss - d_loss)
        # joint_loss = d_loss
        # if min_joint_loss > joint_loss:
        #     joint_loss_improvement_ctr = 0
        #     min_joint_loss = joint_loss
        # d_model.save_weights(saved_models_path + 'discriminator_weights.h5')
        # g_model.save_weights(saved_models_path + 'generator_weights.h5')
        # d_model.save(saved_models_path + 'discriminator')
        # g_model.save(saved_models_path + 'generator')
        # else:
        #     joint_loss_improvement_ctr += 1
        #     if joint_loss_improvement_ctr == early_stop:
        #         print()
        #         return history

        # history['d_loss'].append(d_loss)
        # history['g_loss'].append(g_loss)
        print()
    # d_model.save(saved_models_path + 'discriminator')
    # g_model.save(saved_models_path + 'generator')
    d_model.save_weights(saved_models_path + 'discriminator_weights.h5')
    g_model.save_weights(saved_models_path + 'generator_weights.h5')
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
    # discriminator = define_discriminator(output_shape)
    discriminator = define_deep_discriminator(output_shape)
    # discriminator = define_discriminator2(output_shape)
    # discriminator.summary()

    # create the generator
    generator = define_generator(noise_dim, output_shape)
    noise_dim = (10, 10, 1)
    generator = define_deep_generator(noise_dim, output_shape)
    # generator.summary()

    # create the gan
    gan_model = define_gan(generator, discriminator)

    if action is "train":
        # Training the GAN model.
        history = train(normed_data, meta, generator, discriminator, gan_model, noise_dim,
                        epochs=20, batch_size=128, early_stop=100)
        plot_history(history, y_scale="linear")
        # plot_history(history, y_scale="log")

    if action is "eval":
        # load models
        discriminator.load_weights(saved_models_path + 'discriminator_weights.h5')
        generator.load_weights(saved_models_path + 'generator_weights.h5')
        # discriminator = tf.keras.models.load_model(saved_models_path + 'discriminator')
        # generator = tf.keras.models.load_model(saved_models_path + 'generator')

        x_fake, y_fake = generate_fake_samples(generator, noise_dim, 100)
        prediction = discriminator.predict(x_fake)
        success_num = np.count_nonzero(prediction > 0.5)
        fake_samples = x_fake * (max_data - min_data) + min_data
        filter_predictions = prediction < 0.5
        filter_samples = []
        filter_x = []
        for i in range(len(prediction)):
            if filter_predictions[i]:
                filter_samples.append(fake_samples[i])
                filter_x.append(x_fake[i])
        filter_samples = np.array(filter_samples)
        success_predictions = prediction[filter_predictions]
        # max_value_index = np.argmax(prediction)
        samples_data = data_back_transformation(filter_samples, meta)
        for i in range(len(filter_samples)):
            # min_dist_line, min_dist = find_min_dist(filter_samples[i], numpy_data)
            min_dist_line, min_dist = find_min_dist(filter_x[i], normed_data)
            min_dist = Decimal(min_dist)
            min_dist = round(min_dist, 3)
            samples_data[i].append(success_predictions[i])
            samples_data[i].append(min_dist)
            samples_data[i].append(min_dist_line)
        headline = []
        for attr in meta:
            headline.append(attr)
        headline.append("prediction")
        headline.append("min normal dist")
        headline.append("min dist line")
        samples_data.insert(0, headline)
        with open("failed_samples.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(samples_data)


def plot_confidence(y_hat):
    # idx_spam = np.where(y_test == 1)[0]
    # idx_ham = np.where(y_test == 0)[0]
    plt.hist(y_hat[:, 1], histtype='step', label='class 1')
    # plt.hist(y_hat[idx_spam, 1], histtype='step', label='class 1')
    # plt.hist(y_hat[idx_ham, 1], histtype='step', label='class 0')
    plt.xlabel('Prediction')
    plt.ylabel('Number of observations')
    plt.legend(loc='upper left')
    plt.show()


def confidence_loss(random_forest, y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    # return bce(y_true, y_pred[:, -1])
    # return backend.mean(y_true)
    y_pred_samples = y_pred[:, :-1]
    y_hat = random_forest.predict_proba(y_pred_samples)
    y_pred_confidences = y_hat[:, 1]
    y_pred_confidences = y_pred_confidences.reshape((len(y_pred_confidences), 1))
    return bce(y_true, y_pred_confidences)
    # return backend.sum(backend.abs(y_true - y_pred[:, -1]))


def define_generator_for_random_forest(noise_dim, output_shape, desired_confidence_dim):
    # Define the tensors for the two input images
    noise_input = Input(shape=(noise_dim,))
    desired_confidence_input = Input(shape=(desired_confidence_dim,))

    x = Concatenate()([noise_input, desired_confidence_input])
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)

    x = Concatenate()([x, desired_confidence_input])
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)

    x = Concatenate()([x, desired_confidence_input])
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(output_shape, activation='sigmoid')(x)
    # x = Concatenate()([x, desired_confidence_input])
    model = Model(inputs=[noise_input, desired_confidence_input], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def define_generator_for_random_forest2(noise_dim, output_shape, desired_confidence_dim):
    # Define the tensors for the two input images
    # noise_input = Input(shape=noise_dim)
    noise_input = Input(shape=(noise_dim,))
    desired_confidence_input = Input(shape=(desired_confidence_dim,))

    x = Concatenate()([noise_input, desired_confidence_input])
    x = Dense(100)(x)
    x = Reshape((10, 10, 1))(x)

    x = Conv2DTranspose(filters=32, kernel_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(filters=16, kernel_size=(4, 4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(filters=8, kernel_size=(8, 8))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(output_shape, activation='sigmoid')(x)

    x = Concatenate()([x, desired_confidence_input])
    x = Dropout(0.5)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=[noise_input, desired_confidence_input], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def define_generator_for_random_forest3(noise_dim, output_shape, desired_confidence_dim):
    # Define the tensors for the two input images
    # noise_input = Input(shape=noise_dim)
    noise_input = Input(shape=(noise_dim))
    desired_confidence_input = Input(shape=(desired_confidence_dim))

    noise_model = define_deep_generator(noise_dim, output_shape)
    desired_confidence_model = define_deep_generator(desired_confidence_dim, output_shape)

    noise_layer = noise_model(noise_input)
    dc_layer = desired_confidence_model(desired_confidence_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([noise_layer, dc_layer])
    L1_distance = Dropout(0.4)(L1_distance)

    output = Dense(output_shape, activation='sigmoid')(L1_distance)

    model = Model(inputs=[noise_input, desired_confidence_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_sample_from_random_forest(normed_data_x, desired_confidence, y_hat):
    confidences = y_hat[:, 1]
    idx = (np.abs(confidences - desired_confidence)).argmin()
    return normed_data_x[idx], confidences[idx]


def get_desired_confidence_samples_by_random_forest(normed_data_x, desired_confidences, generator_samples_y, y_hat):
    confidences = y_hat[:, 1]
    desired_confidence_samples = []
    for i in range(len(desired_confidences)):
        desired_confidence = desired_confidences[i]
        idx = (np.abs(confidences - desired_confidence)).argmin()
        sample = normed_data_x[idx]
        sample = np.append(sample, generator_samples_y[i])
        desired_confidence_samples.append(sample)
    desired_confidence_samples = np.array(desired_confidence_samples)
    return desired_confidence_samples


def training_step(batch_size, desired_confidence_dim, generator, history, i, iterations, noise_dim, normed_data,
                  normed_data_x, random_forest):
    # generate noise as input for the generator
    # noise = randn(batch_size * noise_dim)
    # noise = noise.reshape(batch_size, noise_dim)
    noise = randn(noise_dim[0] * noise_dim[1] * noise_dim[2] * batch_size)
    # noise = noise.reshape((batch_size, noise_dim[0], noise_dim[1], noise_dim[2]))
    noise = noise.reshape((batch_size, noise_dim[0] * noise_dim[1] * noise_dim[2]))
    # get batch of real samples(y_true) and get their desired confidence
    start_idx = i * batch_size
    remaining_samples = len(normed_data) - start_idx
    end_idx = start_idx + batch_size
    if remaining_samples < batch_size:
        end_idx = start_idx + remaining_samples
        noise = noise[:remaining_samples]
    y_true = normed_data_x[start_idx:end_idx]
    desired_confidence = random_forest.predict_proba(y_true)[:, 1]
    # desired_confidence = np.random.uniform(0, 1, (batch_size, 1))
    full_desired_confidence = []
    for j in range(len(desired_confidence)):
        full_desired_confidence.append(np.full(desired_confidence_dim, desired_confidence[j]))
    full_desired_confidence = np.array(full_desired_confidence)
    # generator_input = [noise, desired_confidence]
    generator_input = [noise, full_desired_confidence]
    # create prediction from the random forest
    generator_samples = generator.predict(generator_input)
    generator_samples_x = generator_samples[:, :-1]
    generator_samples_y = generator_samples[:, -1]
    generator_samples_y = np.reshape(generator_samples_y, (len(y_true), 1))
    # y_true = get_desired_confidence_samples_by_random_forest(normed_data_x, desired_confidence,
    #                                                          generator_samples_y, y_hat)
    y_true = np.hstack((y_true, generator_samples_y))
    # update the generator
    g_loss = generator.train_on_batch(generator_input, y_true)
    history.append(g_loss)
    print_progress(iterations, i + 1, 0, g_loss, batch_size, len(normed_data))


def training_step2(batch_size, generator, history, i, iterations, noise_dim, normed_data, random_forest):
    noise = randn(noise_dim[0] * noise_dim[1] * noise_dim[2] * batch_size)
    noise = noise.reshape((batch_size, noise_dim[0] * noise_dim[1] * noise_dim[2]))
    start_idx = i * batch_size
    remaining_samples = len(normed_data) - start_idx
    end_idx = start_idx + batch_size
    if remaining_samples < batch_size:
        end_idx = start_idx + remaining_samples
        noise = noise[:remaining_samples]
    desired_confidence = np.random.uniform(0, 1, (batch_size, 1))
    generator_input = [noise, desired_confidence]

    # update the generator
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    with tf.GradientTape() as tape:
        # Forward pass.
        logits = generator.predict(generator_input)[:, :-1]
        confidence_pred = random_forest.predict_proba(logits)[:, 1]
        confidence_pred = np.reshape(confidence_pred, (batch_size, 1))
        # Loss value for this batch.
        g_loss = loss_fn(desired_confidence, confidence_pred)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(g_loss, generator.trainable_weights)

    optimizer.apply_gradients(zip(gradients, generator.trainable_weights))
    history.append(g_loss)
    print_progress(iterations, i + 1, 0, g_loss, batch_size, len(normed_data))


def train_generator(generator, normed_data, noise_dim, desired_confidence_dim, random_forest, epochs, batch_size):
    history = []
    normed_data_x = normed_data[:, :-1]
    # y_hat = random_forest.predict_proba(normed_data_x)
    iterations = int(len(normed_data) / batch_size)
    if len(normed_data) % batch_size != 0:
        iterations = iterations + 1
    # manually enumerate epochs
    for epoch in range(epochs):
        print("Epoch (%d/%d)" % (epoch + 1, epochs))
        for i in range(iterations):
            training_step(batch_size, desired_confidence_dim, generator, history, i, iterations, noise_dim, normed_data, normed_data_x, random_forest)
            # training_step2(batch_size, generator, history, i, iterations, noise_dim, normed_data)
        print()
    generator.save_weights(saved_models_path + 'generator_weights.h5')
    return history


def eval_random_forest(random_forest):
    with open(saved_models_path + test_file_name, 'rb') as f:
        x_test, y_test = pickle.load(f)
    y_pred = random_forest.predict(x_test)
    y_hat = random_forest.predict_proba(x_test)
    plot_confidence(y_hat)
    y_confidence = np.empty(len(y_pred))
    for i in range(len(y_pred)):
        prediction = int(y_pred[i])
        confidence = y_hat[i][prediction]
        y_confidence[i] = confidence
    y_confidence = y_hat[:, 1]
    min_confidence = y_confidence.min(0)
    max_confidence = y_confidence.max(0)
    mean_confidence = y_confidence.mean(0)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("min_confidence: %f, max_confidence: %f, mean_confidence: %f" % (min_confidence, max_confidence,
                                                                           mean_confidence))


def eval_generator(generator, random_forest, noise_dim, generated_size):
    noise = randn(noise_dim * generated_size)
    noise = noise.reshape((generated_size, noise_dim))
    desired_confidence = np.random.uniform(0, 1, (generated_size, 1))
    desired_confidence = np.sort(desired_confidence, axis=0)

    # generate samples
    generator_input = [noise, desired_confidence]
    generator_samples = generator.predict(generator_input)
    generator_samples_x = generator_samples[:, :-1]

    # eval samples
    y_hat = random_forest.predict_proba(generator_samples_x)
    plot_confidence(y_hat)

    random_forest_confidence = y_hat[:, 1]
    random_forest_confidence = np.reshape(random_forest_confidence, generated_size)
    desired_confidence = np.reshape(desired_confidence, generated_size)
    difference_confidence = np.abs(desired_confidence - random_forest_confidence)

    bucket_difference_confidence = []
    bucket_max = 0.1
    difference_sum = 0
    difference_count = 0
    for i in range(len(desired_confidence)):
        if desired_confidence[i] < bucket_max:
            difference_count += 1
            difference_sum += difference_confidence[i]
        else:
            bucket_difference_confidence.append(difference_sum/difference_count)
            bucket_max += 0.1
            difference_count = 1
            difference_sum = difference_confidence[i]
    bucket_difference_confidence.append(difference_sum / difference_count)

    plt.plot(bucket_difference_confidence)
    plt.title('average difference per confidence buckets')
    plt.ylabel('absolute difference')
    plt.xlabel('confidence input')
    plt.xticks(np.arange(0, 10), labels=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1'])
    plt.show()


def part2(action):
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

    # split data to x and y
    x = normed_data[:, :-1]
    y = normed_data[:, -1]

    # noise_dim = 100
    noise_dim = (10, 10, 1)
    output_shape = len(normed_data[0])

    desired_confidence_dim = 1
    # desired_confidence_dim = output_shape
    # desired_confidence_dim = noise_dim

    generator = define_generator_for_random_forest(100, output_shape, desired_confidence_dim)
    # generator = define_generator_for_random_forest2(100, output_shape, desired_confidence_dim)
    # generator = define_generator_for_random_forest3(noise_dim, output_shape, desired_confidence_dim)

    # desired_confidence = 0.3
    # sample, confidence = get_sample_from_random_forest(x, desired_confidence)

    if action is "train":
        # Split dataset into training set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)  # 90% training and 10% test
        # save test data
        with open(saved_models_path + test_file_name, 'wb') as f:
            pickle.dump([x_test, y_test], f)
        # train random forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(x_train, y_train)
        # save the model to disk
        pickle.dump(random_forest, open(saved_models_path + random_forest_filename, 'wb'))
        history = train_generator(generator, normed_data, noise_dim, desired_confidence_dim, random_forest,
                                  epochs=5, batch_size=128)
        plt.plot(history)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.show()

    if action is "eval":
        # load test data
        random_forest = pickle.load(open(saved_models_path + random_forest_filename, 'rb'))
        generator.load_weights(saved_models_path + 'generator_weights.h5')

        eval_random_forest(random_forest)
        eval_generator(generator, random_forest, noise_dim=100, generated_size=1000)


def main():
    # part1(action="train")
    # part1(action="eval")

    # part2(action="train")
    part2(action="eval")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
