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
from mapel.core.persistence import experiment_imports as imports
from mapel.tournaments.objects.GraphSimilarity import Distances, ged_blp

# Make interrupt work with plots
signal.signal(signal.SIGINT, signal.SIG_DFL)

# COUNT = [1, 1, 1, 2, 4, 12, 56, 456, 6880, 191536, 9733056, 903753248]
import builtins
from inspect import getframeinfo, stack

original_print = print


def print_wrap(*args, **kwargs):
    caller = getframeinfo(stack()[1][0])
    original_print("FN:", caller.filename, "Line:", caller.lineno, "Func:", caller.function, ":::", *args,
                   **kwargs)


builtins.print = print_wrap

# read n from args
n = int(os.sys.argv[1])
ex = mt.TournamentExperiment(experiment_id=f'full-non-isomorphic-{n}', distance_id=Distances.GED_BLP)
print(len(ex.distances))

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
ex.distances, _, _, _ = imports.add_distances_to_experiment(ex)
ex.compute_distances(Distances.GED_BLP, parallel=True)
ex.embed(embedding_id='mds')
ex.print_map()
