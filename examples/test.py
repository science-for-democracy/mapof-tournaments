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


# builtins.print = print_wrap

# read n from args
for n in [8]:
    # for distance_id in [Distances.SP, Distances.SPH]:
    for distance_id in [Distances.SPHR]:
        ex = mt.TournamentExperiment(experiment_id=f'full-non-isomorphic-{n}', distance_id=distance_id)
        # ex.save_tournament_plots(path=f"graphs/{n}")
        # ex.distances, _, _, _ = imports.add_distances_to_experiment(ex)
        ex.compute_distances(distance_id, parallel=True, clean=False)
        embedding_id = 'mds'
        ex.embed_2d(embedding_id=embedding_id)
        ex.print_map(show=True,
                     saveas=f"{ex.experiment_id}-{distance_id}-{embedding_id}.png",
                     title=f"Distance: {distance_id}, Embedding: {embedding_id}")
