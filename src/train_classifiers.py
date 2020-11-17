"""
Train Accept/Reject Classifiers
"""

import numpy as np
import argparse
from sklearn import datasets, preprocessing, model_selection
from sklearn import linear_model, svm, neural_network, ensemble, neighbors, naive_bayes
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, plot_roc_curve, \
    plot_precision_recall_curve
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from mlxtend.plotting import plot_confusion_matrix

np.random.seed(229)


def logreg(train_x, train_y, args=None):
    if args.load_dir:
        print("Loading LogisticRegression...")
        clf = load(os.path.join(args.load_dir, "logreg.joblib"))
        return clf

    print("Training LogisticRegression...")
    if args.no_search:
        clf = linear_model.LogisticRegression(max_iter=10000)
        clf.fit(train_x, train_y)
        return clf

    # GridSearch for Logistic Regression
    param_grid = [
        {"C": [0.1, 0.2, 0.3, 0.4, 1.0], "solver": ["lbfgs"], "penalty": ["l2"]},
        {"C": [0.01, 0.1, 0.2, 0.3, 0.4, 1.0], "solver": ["liblinear"], "penalty": ["l1", "l2"]},
        {"C": [0.01, 0.1, 0.2, 0.3, 0.4, 1.0], "solver": ["liblinear"], "penalty": ["l2"], "dual": [True]},
    ]
    clf = linear_model.LogisticRegression(max_iter=10000)
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) LogisticRegression", cv.best_params_)
    print("(best cross validation accuracy) LogisticRegression", cv.best_score_)
    return cv.best_estimator_


def mlp(train_x, train_y, args=None):
    if args.load_dir:
        print("Loading MLPClassifier...")
        clf = load(os.path.join(args.load_dir, "mlp.joblib"))
        return clf

    print("Training MLPClassifier...")
    if args.no_search:
        clf = neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(32,))
        clf.fit(train_x, train_y)
        return clf

    param_grid = [
        {
            "hidden_layer_sizes": [(4,), (32,)],
            "activation": ["tanh", "relu"],
            "solver": ["adam"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "max_iter": [3000, 5000],
            "learning_rate": ["constant", "adaptive"]
        }
    ]

    clf = neural_network.MLPClassifier()
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) MLP", cv.best_params_)
    print("(best cross validation accuracy) MLP", cv.best_score_)
    return cv.best_estimator_


def knn(train_x, train_y, args=None):
    print("Training KNN...")
    if args.no_search:
        clf = neighbors.KNeighborsClassifier(n_neighbors=2)
        clf.fit(train_x, train_y)
        return clf

    param_grid = [
        {"n_neighbors": [2, 3, 5, 10]}
    ]

    clf = neighbors.KNeighborsClassifier()
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) KNN", cv.best_params_)
    print("(best cross validation accuracy) KNN", cv.best_score_)
    return cv.best_estimator_


def sgd_clf(train_x, train_y, args=None):
    if args.load_dir:
        print("Loading SGDClassifier...")
        clf = load(os.path.join(args.load_dir, "sgd_clf.joblib"))
        return clf

    print("Training SGDClassifier...")
    if args.no_search:
        clf = linear_model.SGDClassifier()
        clf.fit(train_x, train_y)
        return clf

    param_grid = [
        {"alpha": [0.01, 0.0001]}
    ]
    clf = linear_model.SGDClassifier()
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) SGDClassifier", cv.best_params_)
    print("(best cross validation accuracy) SGDClassifier", cv.best_score_)
    return cv.best_estimator_


def random_forest(train_x, train_y, args=None):
    if args.load_dir:
        print("Loading RandomForestClassifier...")
        clf = load(os.path.join(args.load_dir, "random_forest.joblib"))
        return clf

    print("Training RandomForestClassifier...")
    if args.no_search:
        clf = ensemble.RandomForestClassifier()
        clf.fit(train_x, train_y)
        return clf

    param_grid = [
        {"n_estimators": [100, 200]}
    ]
    clf = ensemble.RandomForestClassifier()
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) RandomForestClassifier", cv.best_params_)
    print("(best cross validation accuracy) RandomForestClassifier", cv.best_score_)
    return cv.best_estimator_


def adaboost(train_x, train_y, args=None):
    if args.load_dir:
        print("Loading AdaBoostClassifier...")
        clf = load(os.path.join(args.load_dir, "adaboost.joblib"))
        return clf

    print("Training AdaBoostClassifier...")
    if args.no_search:
        clf = ensemble.AdaBoostClassifier()
        clf.fit(train_x, train_y)
        return clf

    param_grid = [
        {"n_estimators": [50, 200], "learning_rate": [1.0, 0.5]}
    ]

    clf = ensemble.AdaBoostClassifier()
    cv = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    cv.fit(train_x, train_y)

    print("(best params) AdaBoostClassifier", cv.best_params_)
    print("(best cross validation accuracy) AdaBoostClassifier", cv.best_score_)
    return cv.best_estimator_


def reject_all(train_x, train_y, test_x, test_y):
    print("============= {}-{} =============".format("Reject-ALL", "Reject-ALL"))
    # get the metrics for reject all
    test_y_hat = np.zeros(test_y.shape)
    test_score = 100.0 * sum(test_y == test_y_hat) / len(test_y_hat)
    print('Test accuracy: {} in {} examples'.format(test_score, len(test_y_hat)))
    # per-label precision/recall/f1
    scores = precision_recall_fscore_support(test_y, test_y_hat, average=None, labels=[0, 1])
    print("Precision/Recall/F1:", scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", help="Path to all feature folders", type=str, default="../features/dir")
    parser.add_argument("--out_dir", help="Path to outputs", type=str, default="./out")
    parser.add_argument("--load_dir", help="Path to saved models", type=str, default="")
    parser.add_argument("--scale", default=False, action="store_true")
    parser.add_argument("--no_search", default=False, action="store_true")

    args = parser.parse_args()

    train_npz = np.load(os.path.join(args.feature_path, 'train.npz'))
    val_npz = np.load(os.path.join(args.feature_path, 'dev.npz'))
    test_npz = np.load(os.path.join(args.feature_path, 'test.npz'))

    train_x, train_y = train_npz['x'], train_npz['y']
    val_x, val_y = val_npz['x'], val_npz['y']

    # merge train/val
    train_x = np.concatenate((train_x, val_x), axis=0)
    train_y = np.concatenate((train_y, val_y), axis=0)

    test_x, test_y = test_npz['x'], test_npz['y']

    # scale
    if args.scale:
        train_x = preprocessing.scale(train_x)
        test_x = preprocessing.scale(test_x)

    reject_all(train_x, train_y, test_x, test_y)

    classifiers = {
        "logreg": logreg(train_x, train_y, args),
        "sgd_clf": sgd_clf(train_x, train_y, args),
        "mlp": mlp(train_x, train_y, args),
        "knn": knn(train_x, train_y, args),
        "random_forest": random_forest(train_x, train_y, args),
        "adaboost": adaboost(train_x, train_y, args)
    }

    # add ensemble using all best classifiers
    estimators_hard = [(clf_name, clf) for clf_name, clf in classifiers.items()]
    voting_clf_hard = ensemble.VotingClassifier(estimators_hard, voting="hard")
    print("Training Ensemble (hard)...")
    voting_clf_hard.fit(train_x, train_y)
    classifiers["ensemble_hard"] = voting_clf_hard

    # plots
    fig_roc = plt.figure()
    ax_roc = fig_roc.add_subplot(111)
    fig_prc = plt.figure()
    ax_prc = fig_prc.add_subplot(111)

    for clf_name, clf in classifiers.items():
        print("============= {}-{} =============".format(clf_name, clf))
        train_y_hat = clf.predict(train_x)
        train_score = 100.0 * sum(train_y == train_y_hat) / len(train_y_hat)
        print('Train accuracy: {} in {} examples'.format(train_score, len(train_y_hat)))
        test_y_hat = clf.predict(test_x)
        test_score = 100.0 * sum(test_y == test_y_hat) / len(test_y_hat)
        print('Test accuracy: {} in {} examples'.format(test_score, len(test_y_hat)))
        # per-label precision/recall/f1
        scores = precision_recall_fscore_support(test_y, test_y_hat, average=None, labels=[0, 1])
        print("Precision/Recall/F1:", scores)

        # save best classifiers
        if clf_name not in {"knn", "ensemble_soft", "ensemble_hard"}:
            save_path = os.path.join(args.out_dir, clf_name) + ".joblib"
            dump(clf, save_path)

        # confusion matrix
        confusion_mat = confusion_matrix(test_y, test_y_hat)
        print("Confusion Matrix:", confusion_mat)
        fig_conf, ax = plot_confusion_matrix(conf_mat=confusion_mat)
        fig_conf.savefig(os.path.join(args.out_dir, "{}_CONF_MAT.pdf".format(clf_name)))

        # plot roc curve
        if clf_name not in {"knn", "ensemble_soft", "ensemble_hard", "sgd_clf"}:
            disp_roc = plot_roc_curve(clf, test_x, test_y)
            disp_roc.plot(ax=ax_roc)
            # plot prc curve
            disp_prc = plot_precision_recall_curve(clf, test_x, test_y)
            disp_prc.plot(ax=ax_prc)

    # save plots
    x = np.linspace(0, 1, len(test_y))
    ax_roc.plot(x, x, '--b', linewidth=2)
    fig_roc.suptitle("Receiver Operating Characteristic")
    fig_roc.savefig(os.path.join(args.out_dir, "ROC.pdf"))

    ax_prc.hlines(y=np.sum(test_y) / len(test_y), xmin=0, xmax=1, linestyle='--', color='b')
    fig_prc.suptitle("Precision-Recall Curve")
    fig_prc.savefig(os.path.join(args.out_dir, "PRC.pdf"))


if __name__ == "__main__":
    main()
