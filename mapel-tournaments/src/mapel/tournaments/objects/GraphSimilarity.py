import marshal
import types
import pickle
from copy import deepcopy
from enum import Enum, StrEnum
from functools import partial

import gurobipy as gp
import numpy as np
import networkx as nx
import tqdm
from numpy.ma.core import sqrt
from scipy import stats
from scipy.stats import wasserstein_distance

from mapel.core.inner_distances import emd, emd2, l1
from mapel.core.matchings import solve_matching_vectors


class Distances(StrEnum):
  GED = 'GED'  # Networkx Graph Edit Distance
  GED_OPT = 'GED_OPT'  # Networkx Optimize Graph Edit Distance - approximation
  GED_BLP = 'GED_BLP'  # Graph Edit Distance - BLP
  GED_BLP2 = 'GED_BLP2'  # Graph Edit Distance - Faster BLP
  DEGREE_EMD = 'DEGREE_EMD'  # Degree Earth Mover Distance
  SP = 'SP'  # Shortest Paths
  SPH = 'SPH'  # Shortest Paths Histogram
  SPH2 = 'SPH2'
  SPH_NORM = 'SPH_NORM'  # Shortest Paths Histogram Normalized
  SPHR = 'SPHR'  # Shortest Paths Histogram Restricted
  DEGREE_CEN = "DEGREE_CEN"
  EIGEN_CEN = "EIGEN_CEN"
  KATZ_CEN = "KATZ_CEN"
  CLOSENESS_CEN = "CLOSENESS_CEN"
  BETWEENNESS_CEN = "BETWEENNESS_CEN"
  LOAD_CEN = "LOAD_CEN"
  HARMONIC_CEN = "HARMONIC_CEN"
  # PERCOLATION_CEN = "PERCOLATION_CEN"  # This does not work right now
  # TROPHIC_LEVELS_CEN = "TROPHIC_LEVELS_CEN"  # This doesn't seem to make sense
  # VOTERANK_CEN = "VOTERANK_CEN"  # Same
  LAPLACIAN_CEN = "LAPLACIAN_CEN"
  PAGERANK = "PAGERANK"


def degree_earthmover_distance(u, v):
  """Compute the earthmover distance between the sorted degrees of two graphs."""
  u_degrees = sorted([d for n, d in u.graph.out_degree()])
  v_degrees = sorted([d for n, d in v.graph.out_degree()])
  return emd(u_degrees, v_degrees)


def shortestpathswise_distance(u, v, inner_distance=l1):
  sp1 = u.to_shortest_paths_matrix()
  # sp1.sort(axis=1)
  sp2 = v.to_shortest_paths_matrix()
  # sp2.sort(axis=1)
  n = len(sp1)
  cost_array = [[inner_distance(sp1[i], sp2[j]) for i in range(n)] for j in range(n)]
  return solve_matching_vectors(cost_array)[0]


def shortest_paths_histogram_distance(u, v, inner_distance=emd):
  n = len(u.graph.nodes())
  hist1 = get_shortest_paths_histograms(u)
  hist2 = get_shortest_paths_histograms(v)
  cost_array = [[inner_distance(hist1[i], hist2[j]) for i in range(n)] for j in range(n)]
  return (solve_matching_vectors(cost_array)[0])


higher_wasserstein = None


def change_wasserstein(p):
  global higher_wasserstein
  higher_wasserstein = p


def shortest_paths_histogram_distance2(u, v):
  global higher_wasserstein
  return shortestpathswise_distance(u, v, inner_distance=higher_wasserstein)


# def shortest_paths_histogram_distance_restricted(u, v, inner_distance=emd):
#   n = len(u.graph.nodes())
#   restr = round(sqrt(n))
#   restr = n // 2
#   hist1 = get_shortest_paths_histograms(u)
#   hist1[:, n // restr] = np.sum(hist1[:, n // restr:], axis=1)
#   hist1[:, n // restr + 1:] = 0
#   hist2 = get_shortest_paths_histograms(v)
#   hist2[:, n // restr] = np.sum(hist2[:, n // restr:], axis=1)
#   hist2[:, n // restr + 1:] = 0
#   # print(hist1, hist2)
#   cost_array = [[inner_distance(hist1[i], hist2[j]) for i in range(n)] for j in range(n)]
#   return (solve_matching_vectors(cost_array)[0])
def shortest_paths_histogram_distance_restricted(u, v, inner_distance=emd):
  n = len(u.graph.nodes())
  hist1 = get_shortest_paths_histograms(u)
  hist1[:, -1] = 0
  hist2 = get_shortest_paths_histograms(v)
  hist2[:, -1] = 0
  print(hist2)
  cost_array = [[inner_distance(hist1[i], hist2[j]) for i in range(n)] for j in range(n)]
  return (solve_matching_vectors(cost_array)[0])


def shortest_paths_histogram_distance_normalized(u, v, inner_distance=emd):
  hist1 = get_shortest_paths_histograms(u)
  hist1 = hist1 / hist1.sum(axis=1, keepdims=True)
  hist2 = get_shortest_paths_histograms(v)
  hist2 = hist2 / hist2.sum(axis=1, keepdims=True)
  cost_array = [[inner_distance(hist1[i], hist2[j]) for i in range(len(u.graph.nodes()))]
                for j in range(len(v.graph.nodes()))]
  return solve_matching_vectors(cost_array)[0]


# Helpers
def get_nth_or_last(iterator, n):
  curr = prev = next(iterator, None)
  for _ in range(n):
    prev = curr
    curr = next(iterator, None)
    if curr is None:
      return prev
  return curr


def parallel_runner(args):
  return args[0](*args[1:])


# This is ugly, but multiprocessing does not work with lambda functions
CENTRALITY_FUNCTION = None


def centrality_wrapper(u, v, inner_distance=l1):
  global CENTRALITY_FUNCTION
  u_centrality = np.array(list(sorted(CENTRALITY_FUNCTION(u.graph).values())))
  v_centrality = np.array(list(sorted(CENTRALITY_FUNCTION(v.graph).values())))
  # u_centrality = np.array(list(CENTRALITY_FUNCTION(u.graph).values()))
  # v_centrality = np.array(list(CENTRALITY_FUNCTION(v.graph).values()))
  return inner_distance(u_centrality, v_centrality)


def centrality_measure(fun):
  global CENTRALITY_FUNCTION
  CENTRALITY_FUNCTION = fun
  return centrality_wrapper


# def gm_wrapper(u, v):
#   import gmatch4py as gm
#   ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
#   ged_result = ged.compare([u.graph], [v.graph])
#   return ged.distance(ged_result)


def get_similarity_measure(distance, **kwargs):
  if distance == Distances.GED:
    return nx.graph_edit_distance
  elif distance == Distances.GED_OPT:
    opt = kwargs['opt'] if 'opt' in kwargs else 1
    return lambda u, v: get_nth_or_last(nx.optimize_graph_edit_distance(u, v), opt)
  elif distance == Distances.DEGREE_EMD:
    return degree_earthmover_distance
  elif distance == Distances.SP:
    return shortestpathswise_distance
  elif distance == Distances.SPH:
    return shortest_paths_histogram_distance
  elif distance == Distances.SPH2:
    return shortest_paths_histogram_distance2
  elif distance == Distances.SPH_NORM:
    return shortest_paths_histogram_distance_normalized
  elif distance == Distances.SPHR:
    return shortest_paths_histogram_distance_restricted
  elif distance == Distances.GED_BLP:
    return ged_blp_wrapper
    # return gm_wrapper
  elif distance == Distances.GED_BLP2:
    return ged_blp_faster_wrapper
  elif distance == Distances.DEGREE_CEN:
    return centrality_measure(nx.out_degree_centrality)
  elif distance == Distances.EIGEN_CEN:
    return centrality_measure(nx.eigenvector_centrality_numpy)
  elif distance == Distances.KATZ_CEN:
    return centrality_measure(nx.katz_centrality)
  elif distance == Distances.CLOSENESS_CEN:
    return centrality_measure(nx.closeness_centrality)
  elif distance == Distances.BETWEENNESS_CEN:
    return centrality_measure(nx.betweenness_centrality)
  elif distance == Distances.LOAD_CEN:
    return centrality_measure(nx.load_centrality)
  elif distance == Distances.HARMONIC_CEN:
    return centrality_measure(nx.harmonic_centrality)
  # elif distance == Distances.PERCOLATION_CEN:
  #   return centrality_measure(nx.percolation_centrality)
  # elif distance == Distances.TROPHIC_LEVELS_CEN:
  #   return centrality_measure(nx.trophic_levels)
  # elif distance == Distances.VOTERANK_CEN:
  #   return centrality_measure(nx.voterank)
  elif distance == Distances.LAPLACIAN_CEN:
    return centrality_measure(nx.laplacian_centrality)
  elif distance == Distances.PAGERANK:
    return centrality_measure(nx.pagerank)
  else:
    raise ValueError('Unknown distance measure: {}'.format(distance))


def get_shortest_paths_histograms(u):
  n = len(u.graph.nodes())
  sp = u.to_shortest_paths_matrix()
  hist = np.apply_along_axis(lambda x: np.histogram(x, bins=np.arange(1, n + 2))[0],
                             1,
                             sp)
  return hist


def ged_blp_wrapper(u, v):
  return round(ged_blp(u.graph, v.graph)[0])


def ged_blp_faster_wrapper(u, v):
  return round(ged_blp_faster(u.graph, v.graph)[0])


def ged_blp(u, v):
  """Compute the graph edit distance between two graphs using a binary linear program. """
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  # Map names to indices
  u = nx.convert_node_labels_to_integers(u, label_attribute='old_label')
  v = nx.convert_node_labels_to_integers(v, label_attribute='old_label')

  n = len(u.nodes())
  m = gp.Model('ged_blp', env=env)
  # Vertex matching variables
  x = np.ndarray(shape=(n, n), dtype=object)
  for i in range(n):
    for j in range(n):
      x[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='x_{},{}'.format(i, j))

  # Edge matching variables
  y = np.ndarray(shape=(n, n, n, n), dtype=object)
  for i, j in u.edges():
    for k, l in v.edges():
      y[i, j, k, l] = m.addVar(vtype=gp.GRB.BINARY,
                               name='y_{},{},{},{}'.format(i, j, k, l))

  # Edge reversed variables
  r = np.ndarray(shape=(n, n), dtype=object)
  for i in range(n):
    for j in range(n):
      r[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='r_{},{}'.format(i, j))

  # Objective
  m.setObjective(gp.quicksum(r[i, j] for i in range(n) for j in range(n)),
                 gp.GRB.MINIMIZE)

  # Constraints
  # Vertex matching constraints
  for i in range(n):
    m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

  # Edge matching constraints
  for i, j in u.edges():
    m.addConstr(gp.quicksum(y[i, j, k, l] for (k, l) in v.edges) == 1)
  for k, l in v.edges():
    m.addConstr(gp.quicksum(y[i, j, k, l] for (i, j) in u.edges()) == 1)

  # Topological constraints
  for i, j in u.edges():
    for k, l in v.edges():
      m.addConstr(y[i, j, k, l] <= x[i, k] + x[i, l] * r[i, j])
      m.addConstr(y[i, j, k, l] <= x[j, l] + x[j, k] * r[i, j])

  # Solve
  m.optimize()

  # Print all variables
  # for v in m.getVars():
  #     print('%s %g' % (v.varName, v.x))

  # Get matching
  matching = []
  for i in range(n):
    for j in range(n):
      if x[i, j].x > 0.5:
        matching.append((u.nodes[i]['old_label'], v.nodes[j]['old_label']))
  return m.objVal, matching


def ged_blp_faster(u, v):
  """Compute the graph edit distance between two graphs using a binary linear program. """
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  # Map names to indices
  u = nx.convert_node_labels_to_integers(u, label_attribute='old_label')
  v = nx.convert_node_labels_to_integers(v, label_attribute='old_label')

  n = len(u.nodes())
  m = gp.Model('ged_blp', env=env)
  # Vertex matching variables
  x = np.ndarray(shape=(n, n), dtype=object)
  for i in range(n):
    for j in range(n):
      x[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='x_{},{}'.format(i, j))

  m.setObjective(
      gp.quicksum(x[i, l] * x[j, k] for i, j in u.edges() for k, l in v.edges()),
      gp.GRB.MINIMIZE)

  # Constraints
  # Vertex matching constraints
  for i in range(n):
    m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

  # Solve
  m.optimize()

  # Print all variables
  # for v in m.getVars():
  #     print('%s %g' % (v.varName, v.x))

  # Get matching
  matching = []
  for i in range(n):
    for j in range(n):
      if x[i, j].x > 0.5:
        matching.append((u.nodes[i]['old_label'], v.nodes[j]['old_label']))
  return m.objVal, matching


def degree_centrality(u, v, inner_distance=emd):
  u_degrees = sorted([d for _, d in u.graph.out_degree()])
  v_degrees = sorted([d for _, d in v.graph.out_degree()])
  return inner_distance(u_degrees, v_degrees)


if __name__ == '__main__':
  # Test ged_blp
  from mapel.tournaments.objects.TournamentInstance import TournamentInstance
  t1 = nx.from_numpy_array(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                           create_using=nx.DiGraph)
  t2 = nx.from_numpy_array(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                           create_using=nx.DiGraph)
  print(ged_blp_faster(t1, t2))
