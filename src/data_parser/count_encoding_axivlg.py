import math
import collections
import numpy as np
from scipy import sparse
import naive_bayes as nb
import os
import argparse
import ScienceParseReader
import Paper
import glob
import re
import spacy
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve, \
    roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Get Simple Count Encoding. Naive bayes classification")

    parser.add_argument("--data_dir", help="Name of the paper venue", type=str,
                        default='../../PeerRead/data/arxiv.cs.lg_2007-2017')
    parser.add_argument('--encodingoutput', nargs='?', default='count_encoding/',
                        help="path for npz file containing embeddings")
    parser.add_argument('--modeloutput', nargs='?', default='nbmodel/arxiv.cs.lg_2007-2017.p',
                        help="path for npz file containing embeddings")

    return parser.parse_args()


def preprocess(message, only_char=True, lower=True, stop_remove=True):
    message = re.sub(r'[^\x00-\x7F]+', ' ', message)
    if lower:
        message = message.lower()
    if only_char:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(message)
        message = ' '.join(tokens)
    tokens = word_tokenize(message)
    if stop_remove:
        tokens = [w for w in tokens if w not in stopwords.words('english')]

    # also remove one-length word
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)


def get_words(message):
    """Get the normalized list of words from a message string.

    For normalization, you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    result = []
    words = message.split()
    for word in words:
        result.append(word.lower())

    return result


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.


    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    countdict = {}
    for message in messages:
        words = get_words(message)
        wordset = set()
        for word in words:
            if word not in wordset:
                wordset.add(word)
                if word in countdict:
                    countdict[word] += 1
                else:
                    countdict[word] = 1
    j = 0
    resultdict = {}
    for entry in countdict:
        if countdict[entry] >= 5:
            resultdict[entry] = j
            j += 1
    return resultdict


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """

    result = np.zeros((len(messages), len(word_dictionary)))
    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                result[i][word_dictionary[word]] += 1
    return result


def load_msg_label(data_dir, data_split):
    pdf_dir = os.path.join(data_dir, data_split, 'parsed_pdfs')
    review_dir = os.path.join(data_dir, data_split, 'reviews')

    paper_content_corpus = []
    papers = []
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))

    for paper_json_filename in paper_json_filenames:
        paper = Paper.from_json(paper_json_filename)
        paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, pdf_dir)
        paper_content_corpus.append(paper.SCIENCEPARSE.get_paper_content())
        papers.append(paper)

    print("Total number of papers in {}".format(pdf_dir), len(papers))
    out_x = []
    out_y = []

    for p in papers:
        norm_msg = preprocess(p.get_abstract())

        out_x.append(norm_msg)
        decision = 1. if p.get_accepted() else 0.
        out_y.append(decision)
    return out_x, np.array(out_y)


def main(args):
    data_dir = args.data_dir

    train_paper, train_labels = load_msg_label(data_dir, "train")
    val_paper, val_labels = load_msg_label(data_dir, "dev")
    test_paper, test_labels = load_msg_label(data_dir, "test")

    dictionary = create_dictionary(train_paper)

    print('Size of dictionary: ', len(dictionary))

    train_matrix = transform_text(train_paper, dictionary)
    val_matrix = transform_text(val_paper, dictionary)
    test_matrix = transform_text(test_paper, dictionary)

    if not os.path.isdir(args.encodingoutput):
        os.mkdir(args.encodingoutput)
    train_matrix_sparse = sparse.csr_matrix(train_matrix)
    trainoutput = os.path.join(args.encodingoutput, "train")
    sparse.save_npz(trainoutput, train_matrix_sparse)

    val_matrix_sparse = sparse.csr_matrix(val_matrix)
    devoutput = os.path.join(args.encodingoutput, "dev")
    sparse.save_npz(devoutput, val_matrix_sparse)

    test_matrix_sparse = sparse.csr_matrix(test_matrix)
    testoutput = os.path.join(args.encodingoutput, "test")
    sparse.save_npz(testoutput, test_matrix_sparse)

    naive_bayes_model = nb.fit_naive_bayes_model(train_matrix, train_labels)

    with open(args.modeloutput, 'wb') as fp:
        pickle.dump(naive_bayes_model, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.modeloutput, 'rb') as fp:
        naive_bayes_model = pickle.load(fp)

    # train
    train_predictions, train_prob = nb.predict_from_naive_bayes_model(naive_bayes_model, train_matrix)
    train_score = 100.0 * np.mean(train_predictions == train_labels)
    print('Train accuracy: %.2f in %d examples' % (round(train_score, 3), train_labels.shape[0]))

    # dev
    val_predictions, val_prob = nb.predict_from_naive_bayes_model(naive_bayes_model, val_matrix)
    val_score = 100.0 * np.mean(val_predictions == val_labels)
    print('Dev accuracy: %.2f in %d examples' % (round(val_score, 3), val_labels.shape[0]))

    # test
    test_predictions, test_prob = nb.predict_from_naive_bayes_model(naive_bayes_model, test_matrix)
    test_score = 100.0 * np.mean(test_predictions == test_labels)
    print('Test accuracy: %.2f in %d examples' % (round(test_score, 3), test_labels.shape[0]))

    top_5_words = nb.get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    scores = precision_recall_fscore_support(test_labels, test_predictions, average=None, labels=[0, 1])
    print("Precision/Recall/F1:", scores)
    # confusion matrix
    confusion_mat = confusion_matrix(test_labels, test_predictions)
    print("Confusion Matrix:", confusion_mat)

    # fig_roc = plt.figure()
    # ax_roc = fig_roc.add_subplot(111)
    # fig_prc = plt.figure()
    # ax_prc = fig_prc.add_subplot(111)
    #
    # # plot roc curve
    #
    # fpr, tpr, _ = roc_curve(test_labels, test_prob, pos_label=1)
    # roc_auc = roc_auc_score(test_labels, test_prob)
    # ax_roc.plot(fpr, tpr, color='darkorange',
    #              lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ax_roc.legend()
    # fig_roc.savefig("ICLRNBROC.png")
    #
    # # plot prc curve
    # precision, recall, _ = precision_recall_curve(test_labels, test_prob, pos_label=1)
    # prc_auc = average_precision_score(test_labels, test_prob)
    # ax_prc.plot(recall, precision, color='darkorange',
    #              lw=2, label='PRC curve (area = %0.2f)' % prc_auc)
    # ax_prc.legend()
    # ax_prc.hlines(y=np.sum(test_labels) / len(test_labels), xmin=0, xmax=1, linestyle='--', color='b')
    # fig_prc.savefig("ICLRNBPRC.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
