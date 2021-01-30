from bs4 import BeautifulSoup
import numpy as np
from os import walk
import os
import re
from nltk.corpus import stopwords
from collections import Counter
import string
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow import random as tf_random
from sklearn.cluster import KMeans
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_dictionary(sites, flag_for_autoencoder=True):
    """
    This function takes in the list of data extracted from the websites.
    The data is then passed through a cleaning function to remove punctuation, stopwords,
    and any excess white space.
    Once this cleaning is completed, the data is then converted into a dictionary with mapping:
    dictionary[word] = number of times the word appears in all of the documents
    This data is then used to create the vector of word mappings which will then be applied to each site
    """
    words_list = []
    word_dictionary = {}
    for site in sites:
        words = list(set(get_words(site)))
        for word in words:
            words_list.append(word)
    counts = Counter(words_list)
    if flag_for_autoencoder:
        c = 0
        for word in dict(counts.most_common(150)):
            word_dictionary[word] = c
            c += 1
    else:
        c = 0
        for word in counts:
            if counts[word] >= 10:
                word_dictionary[word] = c
                c += 1
    return word_dictionary


def get_words(site):
    stop = set(stopwords.words('english') + [
        '.', ',', '--', '\'s', '\'n', '?', ')', '(', ':', '\'', '\'re',
        '"', '-', '}', '{', u'—'])
    cleaned = []
    for s in site:
        strings = s.split(' ')
        for a in range(len(strings)):
            strings[a] = strings[a].lower().translate(str.maketrans('', '', string.punctuation))
            strings[a] = re.sub(r'\d+', '', strings[a])
            strings[a] = strings[a].replace('\n', '')
            if strings[a] not in stop and strings[a] != ' ' and strings[a] != '':
                cleaned.append(strings[a])
    return cleaned


def create_matrix(sites, word_dictionary):
    keys = list(word_dictionary.keys())
    matrix_of_words = []
    for site in sites:
        current_vector = np.zeros(len(word_dictionary))
        words = get_words(site)
        for word in words:
            if word in keys:
                index = word_dictionary[word]
                current_vector[index] += 1
        matrix_of_words.append(current_vector)
    return np.asarray(matrix_of_words)


def load_data(path):
    _, _, filenames = next(walk(path))
    files = []
    all_data = []
    for file in filenames:
        f = open(path + file)
        soup = BeautifulSoup(f, features="html.parser")
        site_data = soup.find_all([re.compile(r'^h[1-6]$')])  # , 'title', 'body'])
        result = []
        for i in site_data:
            result.append(i.get_text())
        filename_split = file.split('.')
        for domain in filename_split:
            result.append(domain)
        if result:
            all_data.append(result)
            files.append(file)
    return all_data, files


def test_autoencoder(data):

    y = [0 for _ in range(len(data))]  # this is just for train test split, otherwise it is unused
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    n_inputs = data.shape[1]
    # what should probably happen here is the weights are updated daily or when a new site is added, not every time

    # define encoder
    visible = Input(shape=(n_inputs,))

    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)

    # bottleneck
    n_bottleneck = 10
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # decoder level 2
    d = Dense(n_inputs * 2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)

    # define autoencoder model
    model = Model(inputs=visible, outputs=output)

    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')

    # fit the autoencoder model to reconstruct input
    model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_test, X_test))

    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    return encoder, n_bottleneck


def test_naive_bayes(x, y):
    model = naive_bayes.MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("Naive Bayes Training Accuracy {0}".format(np.mean(train_predictions == y_train)))
    print("Naive Bayes Testing Accuracy {0}".format(np.mean(test_predictions == y_test)))


def test_neural_network(X, y, n_classes):
    """
    X -> Data
    y -> Labels
    here we will do some pseudo NLP for classifying the websites
    """

    n_inputs = X.shape[1]
    y = to_categorical(y, n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

    model = Sequential()
    model.add(Dense(n_inputs*2, input_shape=(n_inputs, )))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_inputs, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_valid, y_valid),
              callbacks=[EarlyStopping('val_loss', patience=5)])
    print("Neural Network Training result: {0}".format(model.evaluate(X_train, y_train, verbose=0)))
    print("Neural Network Testing result: {0}".format(model.evaluate(X_test, y_test, verbose=0)))


if __name__ == "__main__":
    np.random.seed(123)
    tf_random.set_seed(100)
    data, filenames = load_data('data/')
    dictionary = create_dictionary(data, flag_for_autoencoder=True)
    top_features_data = create_matrix(data, dictionary)

    encoding, number_classes = test_autoencoder(top_features_data)
    labels = encoding.predict(top_features_data)
    kmeans = KMeans(n_clusters=10).fit(labels)
    website_labels = kmeans.predict(labels)

    label_counts = Counter(website_labels)
    dictionary = create_dictionary(data, flag_for_autoencoder=False)
    data_matrix = create_matrix(data, dictionary)

    test_naive_bayes(data_matrix, website_labels)
    test_neural_network(data_matrix, website_labels, number_classes)
    print(label_counts)
