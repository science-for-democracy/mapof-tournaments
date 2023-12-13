import itertools
import subprocess
from random import uniform

import networkx as nx
import numpy as np


## Compass tournaments
def ordered(num_participants):
  adjacency_matrix = np.zeros((num_participants, num_participants))
  for i in range(num_participants):
    for j in range(i + 1, num_participants):
      adjacency_matrix[i, j] = 1
  return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)


def rock_paper_scissors(num_participants):
  adjacency_matrix = np.zeros((num_participants, num_participants))
  for jump_length in range(1, num_participants // 2 + 1):
    # For the last iteration with even number of participants, we only set half of the edges.
    for i in range(num_participants if jump_length < (num_participants + 1) //
                   2 else num_participants // 2):
      j = (i + jump_length) % num_participants
      adjacency_matrix[i, j] = 1
  return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)


### Statistical
def uniform_random(num_participants, count):
  """Choose each edge direction with probability 0.5"""
  tournaments = []
  for i in range(count):
    adjacency_matrix = np.zeros((num_participants, num_participants))
    for i in range(num_participants):
      for j in range(i + 1, num_participants):
        if np.random.rand() < 0.5:
          adjacency_matrix[i, j] = 1
        else:
          adjacency_matrix[j, i] = 1
    tournaments.append(nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph))
  return tournaments


def condorcet_noise(num_participants, count, p):
  """Start with an initial ordered tournament and invert each edge with probability p"""
  tournaments = []
  for i in range(count):
    adjacency_matrix = np.zeros((num_participants, num_participants))
    for i in range(num_participants):
      for j in range(i + 1, num_participants):
        if np.random.rand() < p:
          adjacency_matrix[j, i] = 1
        else:
          adjacency_matrix[i, j] = 1
    tournaments.append(nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph))
  return tournaments


### From elections
import mapel.elections.objects.OrdinalElection as oe


def _from_ordinal_election(election):
  """Create a tournament from an ordinal election."""
  n = election.num_candidates
  election.prepare_instance()
  adjacency_matrix = np.zeros((n, n))
  pairwise_matrix = election.votes_to_pairwise_matrix()
  for i in range(n):
    for j in range(i + 1, n):
      if pairwise_matrix[i, j] > pairwise_matrix[j, i]:
        adjacency_matrix[i, j] = 1
      else:
        adjacency_matrix[j, i] = 1
  return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)


def ordinal_election_culture(culture_id, num_participants, count, params):
  """Create tournaments from a registered ordinal election culture."""
  if 'num_voters' not in params:
    raise ValueError(
        "num_voters must be specified for the 'ordinal_election' culture inside 'params'")
  tournaments = []
  for i in range(count):
    election = oe.OrdinalElection(culture_id=culture_id,
                                  num_voters=params['num_voters'],
                                  num_candidates=num_participants,
                                  params=params)
    tournaments.append(_from_ordinal_election(election))
  return tournaments


### Special
def diffused(tournament, count, alpha=None, ev=None):
  """Create a family of tournaments by randomly inverting edges. The
  probability of inverting an edge is alpha. If ev is specified, alpha is
  computed as ev / |E|."""
  if ev is not None:
    alpha = ev / len(tournament.graph.edges())
  if alpha is None:
    raise ValueError("Either alpha or ev must be specified for the 'diffused' culture")
  tournaments = {}
  for i in range(count):
    g = tournament.graph.copy()
    c = 0
    for e in g.edges():
      if uniform(0, 1) < alpha:
        c += 1
        g.remove_edge(*e)
        g.add_edge(*reversed(e))
    tournaments[f"diffused_a={alpha}_c={c}_{i}"] = g
  return tournaments


tournament_count_lookup = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 12,
    6: 56,
    7: 456,
    8: 6880,
    9: 191536,
    10: 9733056,
    11: 903753248,
    12: 154108311168,
    13: 48542114686912,
    14: 28401423719122304,
    15: 31021002160355166848,
    16: 63530415842308265100288,
    17: 244912778438520759443245824,
}


def nauty(n, size, params):
  """Generate size random tournaments with n participants using nauty."""
  args = ["nauty-gentourng", str(n)]
  if 'resmod' in params:
    args.append(params['resmod'])
  elif size < tournament_count_lookup[n]:
    try:
      args.append(f'0/{tournament_count_lookup[n]*2/size}')
    except KeyError:
      raise ValueError(f"n={n} is too big, please supply res/mod manually")
  if 'quiet' in params and params['quiet']:
    args.append('-q')
  proc = subprocess.run(args, stdout=subprocess.PIPE)
  graphs = []
  lines = proc.stdout.decode('utf-8').split('\n')[:-1]
  if len(lines) > size:
    lines = np.random.choice(lines, size, replace=False)
  for line in lines:
    arr = np.zeros((n, n), dtype=np.int8)
    ind = 0
    for i in range(n):
      for j in range(i + 1, n):
        if line[ind] == '1':
          arr[i, j] = 1
        elif line[ind] == '0':
          arr[j, i] = 1
        ind += 1
    graphs.append(nx.from_numpy_array(arr, create_using=nx.DiGraph))
  return graphs


#### TODO: Taken from https://github.com/uschmidtk/MoV/blob/master/experiments.py
import pandas as pd

# def condorcet_noise(m, p):
#   all_edges = []
#   for s in itertools.combinations(range(m), 2):
#     s = list(s)
#     if s[0] != s[1]:
#       coin = np.random.rand()
#       if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
#         all_edges.append((s[0], s[1]))
#       else:
#         all_edges.append((s[1], s[0]))
#   T = nx.DiGraph()
#   T.add_nodes_from(range(m))
#   T.add_edges_from(all_edges)
#   if nx.tournament.is_tournament(T):
#     return T
#   else:
#     print("There has been a mistake, this is not a tournament!")

# def condorcet_noise_pref(n, m, p):
#   all_edges = []

#   for i in range(n):
#     for s in itertools.combinations(range(m), 2):
#       s = list(s)
#       if s[0] != s[1]:
#         coin = np.random.rand()
#         if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
#           all_edges.append((s[0], s[1]))
#         else:
#           all_edges.append((s[1], s[0]))

#   edge_count = pd.Series(all_edges).value_counts()
#   agg_edges = list(edge_count[edge_count > int(n / 2)].index)

#   T = nx.DiGraph()
#   T.add_nodes_from(range(m))
#   T.add_edges_from(agg_edges)
#   if nx.tournament.is_tournament(T):
#     return T
#   else:
#     print("There has been a mistake, this is not a tournament!")


def impartial_culture(n, m):
  all_edges = []
  for i in range(n):
    order = list(np.random.permutation(range(m)))
    for s in itertools.combinations(range(m), 2):
      s = list(s)
      if s[0] != s[1]:
        if (order.index(s[0]) < order.index(s[1])):
          all_edges.append((s[0], s[1]))
        else:
          all_edges.append((s[1], s[0]))

  edge_count = pd.Series(all_edges).value_counts()
  agg_edges = list(edge_count[edge_count > int(n / 2)].index)

  T = nx.DiGraph()
  T.add_nodes_from(range(m))
  T.add_edges_from(agg_edges)
  if nx.tournament.is_tournament(T):
    return T
  else:
    print("There has been a mistake, this is not a tournament!")


def mallows(n, m, phi):
  candmap = {i: i
             for i in range(m)}
  rankmapcounts = pl.gen_mallows(n, candmap, [1], [phi], [list(range(m))])
  all_edges = []
  for i in range(len(rankmapcounts[1])):
    for s in itertools.combinations(range(m), 2):
      if s[0] != s[1]:
        if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
          for j in range(rankmapcounts[1][i]):
            all_edges.append((s[0], s[1]))
        else:
          for j in range(rankmapcounts[1][i]):
            all_edges.append((s[1], s[0]))

  edge_count = pd.Series(all_edges).value_counts()
  agg_edges = list(edge_count[edge_count > int(n / 2)].index)

  T = nx.DiGraph()
  T.add_nodes_from(range(m))
  T.add_edges_from(agg_edges)
  if nx.tournament.is_tournament(T):
    return T
  else:
    print("There has been a mistake, this is not a tournament!")


def urn(n, m, replace):
  candmap = {i: i
             for i in range(m)}
  rankmapcounts = pl.gen_urn_strict(n, replace, candmap)
  #print(rankmapcounts)
  all_edges = []
  for i in range(len(rankmapcounts[1])):
    for s in itertools.combinations(range(m), 2):
      if s[0] != s[1]:
        if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
          for j in range(rankmapcounts[1][i]):
            all_edges.append((s[0], s[1]))
        else:
          for j in range(rankmapcounts[1][i]):
            all_edges.append((s[1], s[0]))

  edge_count = pd.Series(all_edges).value_counts()
  agg_edges = list(edge_count[edge_count > int(n / 2)].index)
  #print(edge_count)

  T = nx.DiGraph()
  T.add_nodes_from(range(m))
  T.add_edges_from(agg_edges)
  if nx.tournament.is_tournament(T):
    return T
  else:
    print("There has been a mistake, this is not a tournament!")


if __name__ == '__main__':
  print(impartial_culture(10, 5))
