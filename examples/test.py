import os
import signal
import pickle

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import triu_indices

import networkx as nx
import mapel.tournaments as mt
from mapel.core import glossary, objects
from mapel.tournaments.objects.GraphSimilarity import Distances, ged_blp

# Make interrupt work with plots
signal.signal(signal.SIGINT, signal.SIG_DFL)

# COUNT = [1, 1, 1, 2, 4, 12, 56, 456, 6880, 191536, 9733056, 903753248]


def get_all_non_isomorphic_graphs(n):
    """Generate all non-isomorphic graphs with n nodes."""
    if os.path.exists(f'pickle/graphs_{n}.pkl'):
        return pickle.load(open(f'pickle/graphs_{n}.pkl', 'rb'))

    def gen(graph, pos):
        if pos >= len(indices[0]):
            yield graph.copy()
            return
        r, c = indices[0][pos], indices[1][pos]
        graph[r, c] = 1
        yield from gen(graph, pos + 1)
        graph[r, c] = 0
        graph[c, r] = 1
        yield from gen(graph, pos + 1)
        graph[c, r] = 0

    indices = triu_indices(n, k=1)
    graphs = []
    for g1 in gen(np.zeros((n, n)), 0):
        g1 = nx.from_numpy_array(g1, create_using=nx.DiGraph())
        for g2 in graphs:
            ged = ged_blp(g1, g2) < 0.5
            iso = nx.is_isomorphic(g1, g2)
            if ged and iso:
                break
            if ged != iso:
                raise Exception('GED and isomorphism are not consistent')
        else:
            graphs.append(g1)
            print(len(graphs))
            # if (len(graphs) == COUNT[n]):
            #     break
    pickle.dump(graphs, open(f'pickle/graphs_{n}.pkl', 'wb'))
    return graphs


ex = mt.TournamentExperiment()
n = 6
for i, g in enumerate(get_all_non_isomorphic_graphs(n)):
    ex.add_family(
        culture_id=f'{i}',
        # color=objects.Experiment.COLORS[i],
        family_id=f'{i}',
        label=f'{i}',
        single=True,
        num_participants=n,
        params={'adjacency_matrix': g})
ex.add_family(
    culture_id='Ordered',
    color='red',
    family_id='Ordered',
    label='Ordered',
    single=True,
    num_participants=n,
    # plot_path='graphs',
    params={'compass': 'ordered'})
ex.add_family(
    culture_id='Rock-Paper-Scissors',
    color='purple',
    family_id='Rock-Paper-Scissors',
    label='Rock-Paper-Scissors',
    single=True,
    num_participants=n,
    # plot_path='graphs',
    params={'compass': 'rock-paper-scissors'})
ex.add_family(
    culture_id='Mixed',
    color='orange',
    family_id='Mixed',
    label='Mixed',
    single=True,
    num_participants=n,
    # plot_path='graphs',
    params={'compass': 'mixed'})
ex.add_family(
    culture_id='Test',
    color='blue',
    family_id='Test',
    label='Test',
    single=True,
    num_participants=n,
    # plot_path='graphs',
    params={'compass': 'test'})
# ex.save_tournament_plots(path=f"graphs/{n}")
ex.compute_distances(Distances.GED_BLP,
                     parallel=True,
                     path=f"distances/full-non-isomorphic_{n}.pickle")
ex.embed(embedding_id='mds')
ex.print_map()
