import ast
import csv
import itertools
import json
import pickle
import os
import subprocess
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas_access as mdb
import pandas as pd
from matplotlib.font_manager import json_dump
from numpy.lib.twodim_base import triu_indices
from progress.bar import Bar
from tqdm.contrib.concurrent import process_map

from mapel.core.objects.Experiment import Experiment
from mapel.core.objects.Family import Family
from mapel.core.objects.Instance import Instance
from mapel.core.utils import make_folder_if_do_not_exist
from mapel.elections.objects.ElectionFamily import ElectionFamily
from mapel.tournaments.objects.GraphSimilarity import (Distances,
                                                       get_similarity_measure,
                                                       parallel_runner,
                                                       ged_blp)
from mapel.tournaments.objects.TournamentInstance import TournamentInstance
from mapel.tournaments.objects.TournamentCultures import nauty


class TournamentFamily(Family):

  def __init__(self,
               culture_id: str = "none",
               family_id='none',
               params: dict = dict(),
               size: int = 1,
               label: str = "none",
               color: str = "black",
               alpha: float = 1.,
               ms: int = 20,
               show=True,
               marker='o',
               starting_from: int = 0,
               path: dict = dict(),
               single: bool = False,
               num_participants=10,
               tournament_ids=None,
               instance_type: str = 'tournament',
               experiment_id: str = 'none') -> None:

    super().__init__(culture_id=culture_id,
                     family_id=family_id,
                     params=params,
                     size=size,
                     label=label,
                     color=color,
                     alpha=alpha,
                     ms=ms,
                     show=show,
                     marker=marker,
                     starting_from=starting_from,
                     path=path,
                     single=single,
                     instance_ids=tournament_ids)

    self.num_participants = num_participants
    self.instance_type = instance_type
    self.experiment_id = experiment_id

  NON_ISO_FILENAME_PREFIX = 'all-non-isomorphic-'

  def get_all_non_isomorphic_graphs(self, n):
    """Generate all non-isomorphic graphs with n nodes."""
    pickle_path = (os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "pickles"))
    if not os.path.exists(pickle_path):
      os.makedirs(pickle_path)
    pickle_file = os.path.join(pickle_path,
                               f'{TournamentFamily.NON_ISO_FILENAME_PREFIX}{n}.pkl')
    if os.path.exists(pickle_file):
      return pickle.load(open(pickle_file, 'rb'))

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
    bar = Bar('Computing distances:', max=int(2**(len(indices[0]))))
    bar.start()
    quickcheck = defaultdict(bool)
    hasht = np.zeros((n, n), dtype=np.longlong)
    for i, (r, c) in enumerate(zip(indices[0], indices[1])):
      hasht[r, c] = 2**(2 * i)
      hasht[c, r] = 2**(2 * i + 1)
    for g1 in gen(np.zeros((n, n)), 0):
      bar.next()
      s = np.sum(g1, axis=1)
      ind = np.argsort(s)
      test = np.sum(g1[ind, :] * hasht)
      if quickcheck[test]:
        continue

      g1 = nx.from_numpy_array(g1, create_using=nx.DiGraph())
      for g2 in graphs:
        if nx.is_isomorphic(g1, g2):
          quickcheck[test] = True
          break
      else:
        graphs.append(g1)
        print(len(graphs))
    pickle.dump(graphs, open(pickle_file, 'wb'))
    return graphs

  def inverse_graph(self, g):
    g_inv = nx.DiGraph()
    g_inv.add_nodes_from(g.nodes)
    for u, v in g.edges:
      g_inv.add_edge(v, u)
    return g_inv

  def prepare_tournament_family(self):
    if self.single:
      if 'adjacency_matrix' in self.params:
        adjacency_matrix = self.params['adjacency_matrix']
        tournaments = {
            self.family_id:
            TournamentInstance(adjacency_matrix,
                               self.experiment_id,
                               self.family_id,
                               self.culture_id)
        }
      elif self.culture_id.startswith('bridge-'):
        if "db" not in self.params:
          raise ValueError('db must be specified for bridge culture.')
        path = os.path.join(os.getcwd(),
                            "experiments",
                            self.experiment_id,
                            "dbs",
                            f'{self.params["db"]}')
        if not os.path.exists(path):
          print(path)
          raise ValueError(f'Database {self.params["db"]} not found.')
        df = mdb.read_table(path, 'Matchdresults', converters_from_schema=False)
        tournaments = {
            self.family_id:
            TournamentInstance.from_bridge_df(df,
                                              self.experiment_id,
                                              self.family_id,
                                              self.culture_id)
        }
      else:
        raise ValueError('Adjacency_matrix must be specified for a single tournament.')
    if 'compass' in self.params:
      tournaments = {
          tournament.instance_id: tournament
          for tournament in TournamentInstance.from_compass(self.num_participants,
                                                            self.params['compass'],
                                                            self.experiment_id,
                                                            self.family_id,
                                                            self.culture_id)
      }
    elif self.culture_id.startswith('json'):
      if "path" not in self.params:
        raise ValueError('path must be specified for json culture.')
      path = os.path.join(os.getcwd(),
                          "experiments",
                          self.experiment_id,
                          "jsons",
                          f'{self.params["path"]}')
      if not os.path.exists(path):
        print(path)
        raise ValueError(f'Json {self.params["path"]} not found.')
      with open(path) as f:
        tournaments = {
            self.family_id:
            TournamentInstance.from_dict_of_lists(json.load(f),
                                                  self.experiment_id,
                                                  self.family_id,
                                                  self.culture_id)
        }
    elif self.culture_id.startswith('nba-'):
      if "dir" not in self.params:
        raise ValueError('dir must be specified for NBA culture.')
      path = os.path.join(os.getcwd(),
                          "experiments",
                          self.experiment_id,
                          f'{self.params["dir"]}')
      tournaments = {}
      files = os.listdir(path) if not path.endswith('.csv') else [path]
      for csv in files:
        if csv.endswith('.csv'):
          df = pd.read_csv(os.path.join(path, csv))
          instance_id = csv.split('_')[0]
          instance = TournamentInstance.from_nba_df(df,
                                                    self.params,
                                                    self.experiment_id,
                                                    instance_id,
                                                    self.culture_id)
          # if len(instance.graph.nodes()) == 30:
          tournaments[instance_id] = instance
          # else:
          #   print(instance_id, len(instance.graph.nodes()))
    elif self.culture_id.startswith('all-non-isomorphic'):
      n = int(self.culture_id.split('-')[-1])
      tournaments = {}
      for i, g in enumerate(self.get_all_non_isomorphic_graphs(n)):
        instance_id = f'{self.family_id}_{i}'
        tournaments[instance_id] = TournamentInstance(g,
                                                      self.experiment_id,
                                                      instance_id,
                                                      self.culture_id)
      tournaments = tournaments
    elif self.culture_id.startswith('inverse-all-non-isomorphic'):
      n = int(self.culture_id.split('-')[-2])
      d = int(self.culture_id.split('-')[-1])
      tournaments = {}
      path = os.path.join(
          os.getcwd(),
          "experiments",
          self.experiment_id,
          "pickles",
          f'inverse-dist-{TournamentFamily.NON_ISO_FILENAME_PREFIX}{n}.pkl')
      if os.path.exists(path):
        inverse_dist = pickle.load(open(path, 'rb'))
      else:
        inverse_dist = {}
      for i, g in enumerate(self.get_all_non_isomorphic_graphs(n)):
        # TournamentInstance(
        #     experiment_id='a', instance_id='a',
        #     graph=self.inverse_graph(g)).save_graph_plot(f"./inverse/{i}.png")
        if i not in inverse_dist:
          print(d, i)
          inverse_dist[i] = ged_blp(g, self.inverse_graph(g))[0]
        if inverse_dist[i] == d:
          instance_id = f'all-non-isomorphic-{n}_{i}'
          tournaments[instance_id] = TournamentInstance(g,
                                                        self.experiment_id,
                                                        instance_id,
                                                        self.culture_id)
      pickle.dump(inverse_dist, open(path, 'wb'))
    elif self.culture_id.startswith('copeland-all-non-isomorphic'):
      n = int(self.culture_id.split('-')[-2])
      d = int(self.culture_id.split('-')[-1])
      tournaments = {}
      for i, g in enumerate(self.get_all_non_isomorphic_graphs(n)):
        if max([deg for _, deg in g.out_degree()]) == d:
          instance_id = f'all-non-isomorphic-{n}_{i}'
          tournaments[instance_id] = TournamentInstance(g,
                                                        self.experiment_id,
                                                        instance_id,
                                                        self.culture_id)
    elif self.culture_id.startswith('nauty'):
      tournaments = {}
      for i, g in enumerate(nauty(self.num_participants, self.size, self.params)):
        instance_id = f'{self.family_id}_{i}'
        tournaments[instance_id] = TournamentInstance(g,
                                                      self.experiment_id,
                                                      instance_id,
                                                      self.culture_id)
    else:
      weights = self.params['weights']
      self.num_participants = len(weights)
      tournaments = {}
      for i in range(self.size):
        instance_id = f'{self.family_id}_{i}'
        tournaments[instance_id] = TournamentInstance.from_weights(
            weights, self.experiment_id, instance_id, self.culture_id)
      tournaments = tournaments
    if 'sample' in self.params and self.params['sample'] == True:
      sample_size = self.num_participants
      sample_count = self.size
      np.random.seed(self.params['seed'] if 'seed' in self.params else 0)
      all_tournaments = list(tournaments.values())
      tournaments = {}
      min_size = min(len(t.graph.nodes) for t in all_tournaments)
      if sample_size > min_size:
        raise ValueError(
            f'Sample size {sample_size} cannot be larger than the size of the smallest tournament in the family {min_size}.'
        )
      for i in range(sample_count):
        sample_id = f'{self.family_id}_sample_{i}'
        tournament = np.random.choice(all_tournaments)
        tournaments[sample_id] = TournamentInstance.sample_tournament(
            tournament, sample_size, self.experiment_id, sample_id, self.culture_id)

    self.instance_ids = list(tournaments.keys())
    return tournaments

  def prepare_from_ordinal_election_family(self):
    if 'num_voters' in self.params:
      num_voters = self.params['num_voters']
      self.params.pop('num_voters')
    else:
      raise ValueError('num_voters must be specified for ordinal election family.')
    election_family = ElectionFamily(culture_id=self.culture_id,
                                     family_id=self.family_id,
                                     params=self.params,
                                     label=self.label,
                                     color=self.color,
                                     alpha=self.alpha,
                                     show=self.show,
                                     size=self.size,
                                     marker=self.marker,
                                     starting_from=self.starting_from,
                                     num_candidates=self.num_participants,
                                     num_voters=num_voters,
                                     path=self.path,
                                     single=self.single,
                                     instance_type=self.instance_type)
    elections = election_family.prepare_family(self.experiment_id)
    tournaments = {}
    for k, v in elections.items():
      tournaments[k] = TournamentInstance.from_election(v)
    self.instance_ids = list(tournaments.keys())
    print(self.instance_ids, "instance_ids")
    return tournaments

  def prepare_family(self, plot_path=None):
    if self.instance_type == 'tournament':
      tournaments = self.prepare_tournament_family()
    elif self.instance_type == 'ordinal':
      tournaments = self.prepare_from_ordinal_election_family()
    else:
      raise ValueError(f'Instance type {self.instance_type} not supported.')
    if plot_path is not None:
      if not os.path.exists(plot_path):
        os.makedirs(plot_path)
      for k, v in tournaments.items():
        v.save_graph_plot(os.path.join(plot_path, str(k)))
    if len(tournaments) != self.size:
      print(
          f"WARNING: Actual size of family {self.family_id} does not match expected size. Expected {self.size}, got {len(tournaments)}."
      )
      self.size = len(tournaments)
    return tournaments
