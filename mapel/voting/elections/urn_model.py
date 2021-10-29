#!/usr/bin/env python

import random as rand

import numpy as np


def generate_urn_votes(num_voters: int = None, num_candidates: int = None,
                       params: dict = None) -> np.ndarray:
    """ Return: ordinal votes from Polya-Eggenberger model_id """

    votes = np.zeros([num_voters, num_candidates])
    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size += params['alpha']

    return votes


def generate_approval_urn_votes(num_voters: int = None, num_candidates: int = None,
                                params: dict = None) -> list:
    """ Return: approval votes from Polya-Eggenberger model_id """

    votes = []
    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            vote = set()
            for c in range(num_candidates):
                if rand.random() <= params['p']:
                    vote.add(c)
            votes.append(vote)
        else:
            votes.append(votes[rand.randint(0, j - 1)])
        urn_size += params['alpha']

    return votes


def generate_approval_truncated_urn_votes(num_voters: int = None, num_candidates: int = None,
                                params: dict = None) -> list:

    ordinal_votes = generate_urn_votes(num_voters=num_voters, num_candidates=num_candidates,
                                       params=params)

    if 'max_range' not in params:
        params['max_range'] = 1.

    votes = []
    k = np.random.randint(low=1., high=int(params['max_range'] * num_candidates))
    for v in range(num_voters):
        votes.append(set(ordinal_votes[v][0:k]))

    return votes

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
