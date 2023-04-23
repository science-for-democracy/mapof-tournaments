import signal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.twodim_base import triu_indices

import networkx as nx
import mapel.tournaments as mt
from mapel.core import glossary, objects

# Make interrupt work with plots
signal.signal(signal.SIGINT, signal.SIG_DFL)


def get_all_non_isomorphic_graphs(n):

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
            if nx.is_isomorphic(g1, g2):
                break
        else:
            graphs.append(g1)
    return graphs


objects.Experiment.COLORS = glossary.COLORS + ['pink', 'yellow', 'blue']

ex = mt.TournamentExperiment()
for i, g in enumerate(get_all_non_isomorphic_graphs(5)):
    fig = plt.figure()
    ex.add_family(culture_id=f'{i}',
                  color=objects.Experiment.COLORS[i],
                  family_id=f'{i}',
                  label=f'{i}',
                  single=True,
                  num_participants=6,
                  params={'adjacency_matrix': g})
    plt.close('all')
ex.compute_distances()
ex.embed(embedding_id='kk')
ex.print_map()
