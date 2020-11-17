import scipy.sparse
import numpy as np
import naive_bayes as nb
import os
import argparse
from joblib import load

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

from sklearn.metrics import precision_recall_curve, roc_curve, \
    roc_auc_score, average_precision_score, plot_roc_curve, \
    plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Get Simple Count Encoding. Naive bayes classification")

    parser.add_argument("--model_dir", help="Name of the paper venue", type=str,
                        default='/home/xiyan/CS229-Final-Project/RESULTS/arxiv_bert')
    parser.add_argument("--nbmodel_dir", help="Name of the paper venue", type=str,
                        default='../RESULTS/arxiv_bow/nbmodel.p')
    parser.add_argument("--feature_path", help="Path to all feature folders", type=str,
                        default="../features/arxiv_lg/bert_abstract_hand")
    parser.add_argument("--nbfeature_path", help="Path to all feature folders", type=str,
                        default="../features/arxiv_lg/bow")
    parser.add_argument("--out_dir", help="Path to outputs", type=str, default="../RESULTS/arxiv_bert")
    return parser.parse_args()


def main(args):
    classifiers = {
        "logreg": "LogReg",
        "sgd_clf": "SVM",
        "mlp": "MLP",
        "knn": "KNN",
        "random_forest": "RandForest",
        "adaboost": "AdaBoost",
        "nb": "NaiveBayes"
    }

    testnb_npz = scipy.sparse.load_npz(os.path.join(args.nbfeature_path, 'test.npz'))
    test_npz = np.load(os.path.join(args.feature_path, 'test.npz'))
    test_x = test_npz['x']
    test_y = test_npz['y']

    test_x = preprocessing.scale(test_x)

    # plots
    fig_roc = plt.figure()
    ax_roc = fig_roc.add_subplot(111)
    fig_prc = plt.figure()
    ax_prc = fig_prc.add_subplot(111)

    for clf_name in classifiers:
        if clf_name != "knn":
            if clf_name == "nb":
                with open(args.nbmodel_dir, 'rb') as fp:
                    clf = pickle.load(fp)
                    print(testnb_npz.todense().shape)
                    nb_test_predict, nb_test_prob = nb.predict_from_naive_bayes_model(clf,
                                                                                      np.asarray(testnb_npz.todense()))
                    # plot roc
                    fpr, tpr, _ = roc_curve(test_y, nb_test_prob, pos_label=1)
                    roc_auc = roc_auc_score(test_y, nb_test_prob)
                    ax_roc.plot(fpr, tpr, color='fuchsia',
                                lw=1, label='NaiveBayes (AUC = %0.2f)' % roc_auc)
                    # ax_roc.legend()

                    # plot prc curve
                    precision, recall, _ = precision_recall_curve(test_y, nb_test_prob, pos_label=1)
                    prc_auc = average_precision_score(test_y, nb_test_prob)

                    ax_prc.plot(recall, precision, color='fuchsia',
                                lw=1, label='NaiveBayes (AP = %0.2f)' % prc_auc)
                    # ax_prc.legend()
            else:
                modelname = clf_name + ".joblib"
                clf = load(os.path.join(args.model_dir, modelname))

                disp_roc = plot_roc_curve(clf, test_x, test_y, name=classifiers[clf_name])
                disp_roc.plot(ax=ax_roc)
                # plot prc curve
                disp_prc = plot_precision_recall_curve(clf, test_x, test_y, name=classifiers[clf_name])
                disp_prc.plot(ax=ax_prc)

    # save plots
    x = np.linspace(0, 1, len(test_y))
    ax_roc.legend()
    ax_roc.plot(x, x, '--b', linewidth=2)
    fig_roc.suptitle("Receiver Operating Characteristic")
    fig_roc.savefig(os.path.join(args.out_dir, "COMBINEDROC.pdf"))
    print("get roc curve")

    ax_prc.hlines(y=np.sum(test_y) / len(test_y), xmin=0, xmax=1, linestyle='--', color='b')
    ax_prc.legend(loc="upper right")
    fig_prc.suptitle("Precision-Recall Curve")
    fig_prc.savefig(os.path.join(args.out_dir, "COMBINEDPRC.pdf"))
    print("get prc curve")


if __name__ == "__main__":
    args = parse_args()
    main(args)
