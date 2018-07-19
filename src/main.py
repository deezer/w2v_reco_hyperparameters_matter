#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
"""

from __future__ import division

import os
import sys
import time
import logging
import argparse
from tqdm import tqdm
from prettytable import PrettyTable

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from gensim.models.word2vec import Word2Vec
from sklearn.neighbors import NearestNeighbors
from data import (SEED, get_data, cold_start)

logger = logging.getLogger(__name__)


def mean_confidence_interval(data, confidence=0.95):
    """ Standard t-test over mean."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h, h


def run(train, test,
        size_embedding, window_size, min_count, workers, it, sample, neg_sample, power_alpha, k):
    """
    """

    # Training modified Word2Vec model.
    model = Word2Vec(train, size=size_embedding,
                     window=window_size, min_count=min_count, workers=workers, sg=1, iter=it, sample=sample,
                     negative=neg_sample, power_alpha=power_alpha)

    # Create a matrix filled with embeddings of all items considered.
    if model.wv.vocab.keys()[0].startswith('track') or model.wv.vocab.keys()[0].startswith('artist'):
        track_elems = [_ for _ in model.wv.vocab.keys() if _.startswith('track')]
    else:
        track_elems = [_ for _ in model.wv.vocab.keys()]
    embedding = [model.wv[elem] for elem in track_elems]
    mapping = {elem: i for i, elem in enumerate(track_elems)}
    mapping_back = {v: k for k, v in mapping.items()}

    # Fit nearest neighbour model.
    neigh = NearestNeighbors()
    neigh.fit(embedding)

    hrk_score = 0.0
    ndcg_score = 0.0
    t = time.time()
    print "Computing scores..."
    for pair_items in tqdm(test):
        emb_0 = embedding[mapping[pair_items[0]]].reshape(1, -1)
        # If the next item and the query item are identical, we consider that the prediction is correct.
        if str(pair_items[1]) == str(pair_items[0]):
             # HR@k
            hrk_score += 1/k
            # NDCG@k
            ndcg_score += 1
        # If the next item and the query item are different,
        # we compute the 10 nearest neighbour and compute the HR@k and NDCG@k.
        else:
            # get neighbors
            emb_neighbors = neigh.kneighbors(emb_0, k+1)[1].flatten()[1:]
            neighbors = [mapping_back[x] for x in emb_neighbors]
            if str(pair_items[1]) in neighbors:
                # HR@k
                hrk_score += 1/k
                # NDCG@k
                # In our case only one item in the retrieved list can be relevant,
                # so in particular the ideal ndcg is 1 and ndcg_at_k = 1/log_2(1+j)
                # where j is the position of the relevant item in the list.
                index_match = (np.where(str(pair_items[1]) == np.array(neighbors)))[0][0]
                ndcg_score += 1/np.log2(np.arange(2, k+2))[index_match]
    hrk_score = hrk_score / len(test)
    ndcg_score = ndcg_score / len(test)

    print "took %1.2f minutes." % ((time.time()-t)/60.)

    return {'HR@%i' % k: 1000*hrk_score, 'NDCG@%i' % k: ndcg_score}

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', dest='path_data', required=True, type=str)
    parser.add_argument('--p2v', help="Evaluate prod2vec", dest='p2v', default=1, type=int)
    parser.add_argument('--size_embedding', dest='size_embedding', default=50, type=int)
    parser.add_argument('--window_size', dest='window_size', default=3, type=int)
    parser.add_argument('--min_count', dest='min_count', default=1, type=int)
    parser.add_argument('--workers', dest='workers', default=10, type=int)
    parser.add_argument('--it', dest='it', default=110, type=int)
    parser.add_argument('--sample', dest='sample', default=0.00001, type=float)
    parser.add_argument('--neg_sample', dest='neg_sample', default=5, type=int)
    parser.add_argument('--power_alpha', dest='power_alpha', default=-0.5, type=float)
    parser.add_argument("--it_conf", help="Number of iterations for confidence intervals", default=10, type=int)
    parser.add_argument('--cold_start', dest='cold_start',  help="Evaluate on cold start", default=-1, type=int)
    parser.add_argument('--k', dest='k',  help="Number of neighbors in nep evaluation", default=10, type=int)
    parser.add_argument('--output', help="Path to folder were to save results.", dest='output', default="", type=str)
    args = parser.parse_args()

    # Get data.
    logger.info("getting data...")
    train_p2v, train_mp2v, _, test = get_data(args.path_data)
    train = train_p2v if args.p2v else train_mp2v
    test = cold_start(train_p2v, test, args.cold_start) if args.cold_start >= 0 else test

    # Name model.
    model_name = '{}_{}_{}_{}_{}_{}_{}'.format(
        "Prod2vec" if args.p2v else "MetaProd2vec",
        "cold_start_{}".format(args.cold_start) if args.cold_start >= 0 else "",
        os.path.split(args.path_data)[-1].split(".")[0],
        args.window_size,
        args.it,
        args.sample,
        args.power_alpha)

    # Several runs to compute confidence interval.
    results = []
    for i in range(args.it_conf):
        result = run(train, test,
                     args.size_embedding, args.window_size, args.min_count, args.workers, args.it,
                     args.sample, args.neg_sample, args.power_alpha, args.k)
        result.update({'Model': model_name, 'result': 'run_{}'.format(i+1)})
        results.append(result)

    hr = 'HR@%i' % args.k
    ndcg = 'NDCG@%i' % args.k
    df = pd.DataFrame.from_records(results, columns=['Model', 'result', hr, ndcg])
    hr_m, _, _, hr_h = mean_confidence_interval(df[hr].tolist(), confidence=0.95)
    ndcg_m, _, _, ndcg_h = mean_confidence_interval(df[ndcg].tolist(), confidence=0.95)
    df.loc[len(df)] = [model_name, "mean", hr_m, ndcg_m]
    df.loc[len(df)] = [model_name, "95%_conf_int", hr_h, ndcg_h]

    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
        table.add_row(row[1:])
    print table.get_string(sortby='result', reversesort=True)

    if args.output:
        df.to_csv(os.path.join(args.output, model_name+".csv"), index=False)
