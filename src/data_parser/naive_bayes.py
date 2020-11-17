import numpy as np


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of the fitted model, consisting of the
    learned model parameters.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    model = {}
    phi_y = np.sum(labels) / labels.shape[0]
    phi_ky = np.zeros((2, matrix.shape[1]))
    messagelength = np.sum(matrix, axis=1)
    spamlength = 0
    nonspamlength = 0
    for i in range(messagelength.shape[0]):
        if labels[i] == 1:
            spamlength += messagelength[i]
        else:
            nonspamlength += messagelength[i]
    vocasize = matrix.shape[1]
    spammessage = matrix[np.where(labels == 1)]
    nonspammessage = matrix[np.where(labels == 0)]
    spamstats = np.sum(spammessage, axis=0)
    nonspamstats = np.sum(nonspammessage, axis=0)
    phi_ky0 = (1 + nonspamstats) / (vocasize + nonspamlength)
    phi_ky1 = (1 + spamstats) / (vocasize + spamlength)
    phi_ky[0] = phi_ky0
    phi_ky[1] = phi_ky1

    model["phi_y"] = phi_y
    model["phi_ky"] = phi_ky
    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model (int array of 0 or 1 values)
    """
    # *** START CODE HERE ***
    log_phi_ky = np.log(model["phi_ky"])
    log_phi_y = np.log(model["phi_y"])
    posterier = np.array([np.log(1 - model["phi_y"]), log_phi_y, ])
    product = np.matmul(matrix, np.transpose(log_phi_ky))
    product = product + posterier
    result = np.zeros(product.shape[0], dtype=int)
    prob = np.zeros(product.shape[0], dtype=float)
    for i in range(product.shape[0]):
        if product[i][1] > product[i][0]:
            result[i] = 1
        prob[i] = 1 / (1 + np.exp(product[i][0] - product[i][1]))

    return result, prob

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_ky0 = model["phi_ky"][0]
    phi_ky1 = model["phi_ky"][1]
    division = np.log(phi_ky1 / phi_ky0)
    increasingindex = np.argsort(division)
    indecies = increasingindex[::-1]
    result = []

    for i in range(5):
        for key in dictionary:
            if indecies[i] == dictionary[key]:
                result.append(key)
    return result

    # *** END CODE HERE ***
