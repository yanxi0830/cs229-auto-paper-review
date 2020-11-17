"""
Extract GloVe + (Hand-Crafted) features, save as npz

> python extract_features.py --data_dir ../../PeerRead/data/iclr_2017/ --data_split test --save_dir ./tmp
"""

import spacy
import csv
import numpy as np
import re
import argparse
import random
import glob
import ScienceParseReader
import Paper
import os
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
import torch


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


def get_hand_features(p):
    hand_features = []
    hand_features.append(p.SCIENCEPARSE.get_most_recent_reference_year())   # 300
    hand_features.append(p.SCIENCEPARSE.get_num_references())               # 301
    hand_features.append(p.SCIENCEPARSE.get_num_refmentions())              # 302
    hand_features.append(p.SCIENCEPARSE.get_avg_length_reference_mention_contexts())    # 303

    hand_features.append(p.abstract_contains_a_term("deep"))    # 304
    hand_features.append(p.abstract_contains_a_term("neural"))  # 305
    hand_features.append(p.abstract_contains_a_term("embedding"))   # 306
    hand_features.append(p.abstract_contains_a_term("outperform"))  # 307
    hand_features.append(p.abstract_contains_a_term("novel"))       # 308
    hand_features.append(p.abstract_contains_a_term("state of the art"))    # 309
    hand_features.append(p.abstract_contains_a_term("state-of-the-art"))    # 310

    # Top 5 indicative words (from ICLR)
    hand_features.append(p.abstract_contains_a_term("discriminator"))   # 311
    hand_features.append(p.abstract_contains_a_term("agent"))           # 312
    hand_features.append(p.abstract_contains_a_term("involves"))        # 313
    hand_features.append(p.abstract_contains_a_term("accelerate"))      # 314
    hand_features.append(p.abstract_contains_a_term("logistic"))        # 315

    hand_features.append(p.SCIENCEPARSE.get_num_recent_references(2017))    # 316
    hand_features.append(p.SCIENCEPARSE.get_num_ref_to_figures())           # 317
    hand_features.append(p.SCIENCEPARSE.get_num_ref_to_tables())            # 318
    hand_features.append(p.SCIENCEPARSE.get_num_ref_to_sections())          # 319
    hand_features.append(p.SCIENCEPARSE.get_num_uniq_words())               # 320
    hand_features.append(p.SCIENCEPARSE.get_num_sections())                 # 322
    hand_features.append(p.SCIENCEPARSE.get_avg_sentence_length())          # 323
    hand_features.append(p.SCIENCEPARSE.get_contains_appendix())            # 324

    hand_features.append(p.get_title_len())                                 # 325
    hand_features.append(p.SCIENCEPARSE.get_num_ref_to_equations())         # 326
    hand_features.append(p.SCIENCEPARSE.get_num_ref_to_theorems())          # 327

    # concat glove + hand
    hand_features = np.array(hand_features)
    return hand_features


def get_bert_features(papers, hand=False):
    """
    Extract pre-trained BERT features
    https://github.com/huggingface/transformers/issues/1950
    """
    out_x = []
    out_y = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    for p in papers:
        # msg = preprocess(p.SCIENCEPARSE.get_paper_content())
        msg = preprocess(p.get_abstract())
        input_ids = torch.tensor(tokenizer.encode("[CLS] " + msg))[:512].unsqueeze(0)
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
        cls_embeddings = last_hidden_states.squeeze()[0]
        features = cls_embeddings.detach().numpy()

        if hand:
            hand_features = get_hand_features(p)
            features = np.concatenate((features, hand_features), axis=0)

        out_x.append(features)

        decision = 1. if p.get_accepted() else 0.
        out_y.append(decision)

    return np.array(out_x), np.array(out_y)


def get_glove_features(papers, hand=True):
    """
    Return numpy (x, y)
    :param papers: List[Paper]
    :param hand: bool, use hand-crafted features
    :return: (train_x, train_y)
    """
    out_x = []
    out_y = []

    nlp = spacy.load('en_core_web_lg')
    for p in papers:
        norm_msg = preprocess(p.get_abstract())
        # norm_msg = preprocess(p.SCIENCEPARSE.get_paper_content())
        glove = nlp(norm_msg).vector

        # HAND FEATURES
        if hand:
            hand_features = get_hand_features(p)
            glove = np.concatenate((glove, hand_features), axis=0)

        out_x.append(glove)
        decision = 1. if p.get_accepted() else 0.
        out_y.append(decision)

    return np.array(out_x), np.array(out_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Name of the paper venue", type=str, default='./data/iclr_2017')
    parser.add_argument("--save_dir", help="Save dir", type=str, default="./features")
    parser.add_argument("--hand", default=False, action="store_true")
    parser.add_argument("--feature_type", help="glove / bert", type=str, default="bert")
    args = parser.parse_args()

    data_dir = args.data_dir

    for split in ["test", "dev", "train"]:
        pdf_dir = os.path.join(data_dir, split, 'parsed_pdfs')
        review_dir = os.path.join(data_dir, split, 'reviews')

        paper_content_corpus = []
        papers = []
        paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))
        for paper_json_filename in paper_json_filenames:
            paper = Paper.from_json(paper_json_filename)
            paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, pdf_dir)
            paper_content_corpus.append(paper.SCIENCEPARSE.get_paper_content())
            papers.append(paper)

        print("Total number of papers in {}".format(pdf_dir), len(papers))

        if args.feature_type == "glove":
            out_x, out_y = get_glove_features(papers, args.hand)
        elif args.feature_type == "bert":
            out_x, out_y = get_bert_features(papers, args.hand)
        else:
            raise NotImplementedError()

        # save features to .npz
        outf = os.path.join(args.save_dir, split) + '.npz'
        np.savez(outf, x=out_x, y=out_y)
        print("Saved to {}!".format(outf), out_x.shape, out_y.shape)


if __name__ == "__main__":
    main()
