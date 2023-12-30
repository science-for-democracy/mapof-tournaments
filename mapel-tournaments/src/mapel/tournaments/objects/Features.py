import time

import matplotlib.pyplot as plt
import networkx as nx
from mapel.tournaments.objects.TournamentCultures import ordered
from mapel.tournaments.objects.TournamentInstance import TournamentInstance
from mapel.tournaments.objects.TournamentSimilarity import ged_blp
from mapel.tournaments.objects.TournamentSolutions import (copeland_winners,
                                                           slater_winner,
                                                           slater_winners,
                                                           top_cycle_winners)

registered_features = {}
aliases = {}


def list_features():
  return registered_features.keys()


def get_feature(feature_id: str):
  """Return the feature function corresponding to the given feature id."""
  if feature_id in aliases:
    return registered_features[aliases[feature_id]]
  else:
    raise ValueError(f"No such feature id: {feature_id}")


def register(name: str | list[str] = [], parallel=False, skip_function_name=False):

  def decorator(func):
    if isinstance(name, str):
      names = [name]
    else:
      names = name
    if isinstance(names, list):
      if not skip_function_name:
        names.append(func.__name__)
      map(lambda n: n.lower(), names)
      registered_features[names[0]] = func
      for n in names:
        aliases[n] = names[0]
      if parallel:
        registered_features[names[0] + "_parallel"] = func
        for n in names:
          aliases[n + "_parallel"] = names[0] + "_parallel"
    else:
      raise ValueError("name must be a string or a list of strings")
    return func

  return decorator


@register("local_transitivity")
def local_transitivity(tournament: TournamentInstance):
  """Calculates how far from a locally transitive graph the graph is, by
  computing ged for each of the two subgraphs induced by each node's in and
  out-degrees.
  """
  g = tournament.graph
  total_distance = 0
  ord_graphs = [ordered(i, 1, {})[0] for i in range(len(g.nodes()))]
  for u in g.nodes():
    g1 = g.subgraph(g.pred[u].keys())
    g2 = g.subgraph(g.succ[u].keys())
    total_distance += ged_blp(ord_graphs[len(g1)], g1)[0]
    total_distance += ged_blp(ord_graphs[len(g2)], g2)[0]
  return total_distance / (len(g.nodes()))


@register("longest_cycle_length")
def longest_cycle_length(tournament):
  """Calculates the longest cycle in the tournament. Uses the fact that
  tournaments always have a Hamiltonian path which is also a cycle if and only
  if the tournament is strongly connected."""
  largest = max(nx.strongly_connected_components(tournament.graph), key=len)
  return len(largest)


@register("slater_winners_count", parallel=True)
def slater_winner_count(tournament):
  """Calculates the number of Slater winners of a given tournament"""
  return len(slater_winners(tournament))


@register("slater_winner_time")
def slater_winner_time(tournament):
  """Calculates the time needed to find a single Slater winner of a given tournament"""
  start = time.time()
  slater_winner(tournament)
  end = time.time()
  return end - start


@register("copeland_winners_count")
def copeland_winner_count(tournament):
  """Calculates the number of Copeland winners of a given tournament"""
  return len(copeland_winners(tournament))


@register("top_cycle_winners_count")
def top_cycle_winner_count(tournament):
  """Calculates the number of top cycle winners of a given tournament"""
  return len(top_cycle_winners(tournament))


@register("highest_degree")
def highest_degree(tournament):
  """Calculates the highest degree of a given tournament"""
  return max(tournament.graph.out_degree(), key=lambda x: x[1])[1]


if __name__ == '__main__':
  t = TournamentInstance.raw(nx.tournament.random_tournament(5))
  # plot and show graph
  print("Longest cycle:", longest_cycle_length(t))
  nx.draw_circular(t.graph, with_labels=True)
  plt.show()
