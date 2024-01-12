import gurobipy as gp
import networkx as nx
import numpy as np
from mapel.core.inner_distances import emd, l1
from mapel.core.matchings import solve_matching_vectors

registered_distances = {}
aliases = {}


def list_distances():
  return registered_distances.keys()


def get_distance(distance_id: str):
  """Return the distance function for the given distance id."""
  if distance_id in aliases:
    return registered_distances[aliases[distance_id]]
  else:
    raise ValueError(f"No such distance id: {distance_id}")


def register(name: str | list[str] = [], skip_function_name=False):

  def decorator(func):
    if isinstance(name, str):
      names = [name]
    else:
      names = name
    if isinstance(names, list):
      if not skip_function_name:
        names.append(func.__name__)
      map(lambda n: n.lower(), names)
      registered_distances[names[0]] = func
      for n in names:
        aliases[n] = names[0]
    else:
      raise ValueError("name must be a string or a list of strings")
    return func

  return decorator


# class Distances(StrEnum):
#   GED = 'GED'  # Networkx Graph Edit Distance
#   GED_OPT = 'GED_OPT'  # Networkx Optimize Graph Edit Distance - approximation
#   GED_BLP = 'GED_BLP'  # Graph Edit Distance - BLP
#   GED_BLP2 = 'GED_BLP2'  # Graph Edit Distance - Faster BLP
#   DEGREE_EMD = 'DEGREE_EMD'  # Degree Earth Mover Distance
#   SP = 'SP'  # Shortest Paths
#   SPH = 'SPH'  # Shortest Paths Histogram
#   SPH2 = 'SPH2'
#   SPH_NORM = 'SPH_NORM'  # Shortest Paths Histogram Normalized
#   SPHR = 'SPHR'  # Shortest Paths Histogram Restricted
#   DEGREE_CEN = "DEGREE_CEN"
#   EIGEN_CEN = "EIGEN_CEN"
#   KATZ_CEN = "KATZ_CEN"
#   CLOSENESS_CEN = "CLOSENESS_CEN"
#   BETWEENNESS_CEN = "BETWEENNESS_CEN"
#   LOAD_CEN = "LOAD_CEN"
#   HARMONIC_CEN = "HARMONIC_CEN"
#   # PERCOLATION_CEN = "PERCOLATION_CEN"  # This does not work right now
#   # TROPHIC_LEVELS_CEN = "TROPHIC_LEVELS_CEN"  # This doesn't seem to make sense
#   # VOTERANK_CEN = "VOTERANK_CEN"  # Same
#   LAPLACIAN_CEN = "LAPLACIAN_CEN"
#   PAGERANK = "PAGERANK"


@register("degree_emd")
def degree_earthmover_distance(u, v):
  """Compute the earthmover distance between the sorted degrees of two graphs."""
  u_degrees = sorted([d for n, d in u.graph.out_degree()])
  v_degrees = sorted([d for n, d in v.graph.out_degree()])
  return emd(u_degrees, v_degrees)


@register("sp")
def shortestpathswise_distance(u, v, inner_distance=l1):
  sp1 = u.to_shortest_paths_matrix()
  # sp1.sort(axis=1)
  sp2 = v.to_shortest_paths_matrix()
  # sp2.sort(axis=1)
  n = len(sp1)
  cost_array = [[inner_distance(sp1[i], sp2[j]) for i in range(n)] for j in range(n)]
  return solve_matching_vectors(cost_array)[0]


@register("sph")
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


@register("sph2")
def shortest_paths_histogram_distance2(u, v):
  global higher_wasserstein
  return shortestpathswise_distance(u, v, inner_distance=higher_wasserstein)


@register("sphr")
def shortest_paths_histogram_distance_restricted(u, v, inner_distance=emd):
  n = len(u.graph.nodes())
  hist1 = get_shortest_paths_histograms(u)
  hist1[:, 2] += np.sum(hist1[:, 3:], axis=1)
  hist1[:, 3:] = 0
  hist2 = get_shortest_paths_histograms(v)
  hist2[:, 2] += np.sum(hist2[:, 3:], axis=1)
  hist2[:, 3:] = 0
  cost_array = [[inner_distance(hist1[i], hist2[j]) for i in range(n)] for j in range(n)]
  return (solve_matching_vectors(cost_array)[0])


@register("sph_norm")
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


def centrality_helper(centrality_function, u, v, inner_distance):
  u_centrality = np.array(list(sorted(centrality_function(u.graph).values())))
  v_centrality = np.array(list(sorted(centrality_function(v.graph).values())))
  return inner_distance(u_centrality, v_centrality)


@register("degree_cen")
def degree_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.out_degree_centrality, u, v, inner_distance)


@register("eigen_cen")
def eigen_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.eigenvector_centrality_numpy, u, v, inner_distance)


@register("katz_cen_test")
def katz_centrality_test(u, v, alpha, inner_distance=l1):
  u_centrality = np.array(
      list(sorted(nx.katz_centrality_numpy(u.graph, alpha=alpha).values())))
  v_centrality = np.array(
      list(sorted(nx.katz_centrality_numpy(v.graph, alpha=alpha).values())))
  return inner_distance(u_centrality, v_centrality)


@register("katz_cen")
def katz_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.katz_centrality_numpy, u, v, inner_distance)


@register("closeness_cen")
def closeness_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.closeness_centrality, u, v, inner_distance)


@register("betweenness_cen")
def betweenness_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.betweenness_centrality, u, v, inner_distance)


@register("load_cen")
def load_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.load_centrality, u, v, inner_distance)


@register("harmonic_cen")
def harmonic_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.harmonic_centrality, u, v, inner_distance)


@register("laplacian_cen")
def laplacian_centrality(u, v, inner_distance=l1):
  return centrality_helper(nx.laplacian_centrality, u, v, inner_distance)


@register("pagerank")
def pagerank(u, v, inner_distance=l1):
  return centrality_helper(nx.pagerank_numpy, u, v, inner_distance)


@register("ged_nx")
def ged_networkx(u, v):
  return nx.graph_edit_distance(u.graph, v.graph)


@register("ged_nx_opt")
def ged_networkx_opt(u, v, opt=1):
  return lambda u, v: get_nth_or_last(nx.optimize_graph_edit_distance(u, v), opt)


def get_shortest_paths_histograms(u):
  n = len(u.graph.nodes())
  sp = u.to_shortest_paths_matrix()
  hist = np.apply_along_axis(lambda x: np.histogram(x, bins=np.arange(1, n + 2))[0],
                             1,
                             sp)
  return hist


@register("ged_blp")
def ged_blp_wrapper(u, v):
  return round(ged_blp(u.graph, v.graph)[0])


@register("ged_blp_parallel")
def ged_blp_parallel_wrapper(u, v):
  return round(ged_blp(u.graph, v.graph, multithreaded=True)[0])


def initial_ged_heuristic(u, v, x):
  u_deg = sorted(u.out_degree(), key=lambda x: x[1])
  v_deg = sorted(v.out_degree(), key=lambda x: x[1])
  l = list(zip(u_deg, v_deg))
  n = len(l)
  for i, j in l[:n // 4] + l[-n // 4:]:
    i, j = i[0], j[0]
    x[i, j].Start = 1
    for k in range(len(u.nodes())):
      if k != i:
        x[k, j].Start = 0
      if k != j:
        x[i, k].Start = 0
  return


class GurobiException(Exception):

  def __init__(self, message, code):
    super().__init__(message)
    self.code = code


def ged_blp(u, v, additional_constraints_func=None, multithreaded=False):
  """Compute the graph edit distance between two graphs using a binary linear program. """
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  # Map names to indices
  u = nx.convert_node_labels_to_integers(u, label_attribute='old_label')
  v = nx.convert_node_labels_to_integers(v, label_attribute='old_label')

  n = len(u.nodes())
  m = gp.Model('ged_blp', env=env)
  if not multithreaded:
    m.setParam(gp.GRB.Param.Threads, 1)

  # Vertex matching variables
  x = np.ndarray(shape=(n, n), dtype=object)
  for i in range(n):
    for j in range(n):
      x[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='x_{},{}'.format(i, j))

  # Heuristic
  # initial_ged_heuristic(u, v, x)

  # Objective
  obj = gp.quicksum(x[i, l] * x[j, k] for i, j in u.edges() for k, l in v.edges())
  m.setObjective(obj, gp.GRB.MINIMIZE)
  m.addConstr(obj <= n * (n - 1) // 4)
  m.addConstr(obj >= 0)

  # Vertex matching constraints
  for i in range(n):
    m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
    m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

  if additional_constraints_func is not None:
    additional_constraints_func(m, x, u, v)

  # Solve
  m.optimize()

  if m.Status != gp.GRB.OPTIMAL:
    raise GurobiException(f"Optimization failed... Status: {m.Status}", m.Status)

  # Get matching
  matching = []
  for i in range(n):
    for j in range(n):
      if x[i, j].x > 0.5:
        matching.append((u.nodes[i]['old_label'], v.nodes[j]['old_label']))
  return m.objVal, matching


@register("ged_blp_old")
def ged_blp_old_wrapper(u, v):
  return round(ged_blp_old(u.graph, v.graph)[0])


def ged_blp_old(u, v):
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


if __name__ == '__main__':
  # Test ged_blp
  t1 = nx.from_numpy_array(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                           create_using=nx.DiGraph)
  t2 = nx.from_numpy_array(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                           create_using=nx.DiGraph)
  print(ged_blp_faster(t1, t2))
