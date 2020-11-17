"""
Train Accept/Reject Classifiers

> python train_classifiers.py
--train_path ./data_parser/tmp/train.npz
--val_path ./data_parser/tmp/dev.npz
--test_path ./data_parser/tmp/test.npz
"""

import numpy as np
import argparse
from sklearn import preprocessing
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

np.random.seed(229)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", help="Path to all feature folders", type=str, default="../features/dir")
    parser.add_argument("--out_dir", help="Path to outputs", type=str, default="./out")
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

    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)

    # only consider hand-features
    train_x = train_x[:, 300:]
    test_x = test_x[:, 300:]
    print(train_x.shape)

    forest = ExtraTreesClassifier(n_estimators=200, random_state=0)

    forest.fit(train_x, train_y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the impurity-based feature importances of the forest
    feature_names = [
        "most_recent_ref_year", "num_refs", "num_ref_mentions", "avg_len_ref_mentions",
        "contains_deep", "contains_neural", "contains_embedding", "contains_outperform", "contains_novel",
        "contains_sota", "contains_s-o-t-a", "contains_discriminator", "contains_agent", "contains_involves",
        "contains_accelerate",
        "contains_logistic", "num_recent_refs", "num_ref_figures", "num_ref_tables", "num_ref_sections",
        "num_uniq_words",
        "num_sections", "avg_sentence_length", "contains_appendix", "title_len", "num_ref_to_eqns", "num_ref_to_thms"
    ]
    feature_names = np.array(feature_names)

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(train_x.shape[1]):
        print("{}. feature {} ({})".format(f + 1, feature_names[indices[f]], importances[indices[f]]))

    plt.rcParams.update({'font.size': 26})

    plt.figure(figsize=(38, 12), dpi=100)
    plt.title("Feature Importance")
    plt.barh(range(train_x.shape[1]), list(reversed(importances[indices])), xerr=std[list(reversed(indices))],
             align="center")
    plt.yticks(range(train_x.shape[1]), labels=feature_names[list(reversed(indices))])
    plt.ylim([-1, train_x.shape[1]])
    plt.xlabel("Gini importance")
    plt.savefig('./feature_importance.pdf')
    plt.show()


if __name__ == "__main__":
    main()
