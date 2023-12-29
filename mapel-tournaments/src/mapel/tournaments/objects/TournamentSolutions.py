import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
from mapel.tournaments.objects.TournamentCultures import ordered
from mapel.tournaments.objects.TournamentInstance import TournamentInstance
from mapel.tournaments.objects.TournamentSimilarity import (GurobiException,
                                                            ged_blp)


def copeland_winners(tournament):
  """
    Returns all Copeland winners of the given tournament.
    """
  copeland_scores = tournament.graph.out_degree()
  winners = list(sorted(copeland_scores, key=lambda x: x[1], reverse=True))
  winners = [winner[0] for winner in winners if winner[1] == winners[0][1]]
  return winners


def slater_winner(tournament):
  """
    Returns one of the Slater winners of the given tournament.
    """
  transitive = ordered(tournament.graph.number_of_nodes(), 0, {})[0]
  _distance, matching = ged_blp(transitive, tournament.graph)
  # print("###slater: ", _distance, matching)
  winner = next(pair[1] for pair in matching if pair[0] == 0)
  return winner


def slater_winners(tournament):
  """Calculates a Slater winner of a given tournament by finding
  the vertex matching with the lowest GED between the input tournament and the
  transitive tournament. After finding a winner, that winner is banned from
  being chosen again and the process is repeated until no more winners can be
  found."""

  def add_constraints(m, x, u, v):
    banned = []
    for i in v.nodes():
      if v.nodes[i]['old_label'] in winners:
        banned.append(i)
    for ban in banned:
      m.addConstr(x[0, ban] == 0)

  g = tournament.graph
  n = len(g.nodes())
  transitive = ordered(len(g.nodes()), 1, {})[0]
  winners = []
  min_distance = n * n
  while True:
    try:
      distance, matching = ged_blp(transitive, g, additional_constraints_func=add_constraints)
      if distance > min_distance:
        break
      min_distance = distance
      winner = next(pair[1] for pair in matching if pair[0] == 0)
      winners.append(winner)
    except GurobiException as e:
      if e.code == gp.GRB.INFEASIBLE:
        break
      else:
        raise e
  return winners


def top_cycle_winners(tournament):
  """Calculates the top cycle winners of a given tournament by first finding
  the copeland set and then expanding it to include all vertices not yet
  dominated by the set."""
  tc = copeland_winners(tournament)
  tcs = set(tc)
  g = tournament.graph
  i = 0
  while i < len(tc):
    for in_edge in g.pred[tc[i]]:
      if in_edge not in tcs:
        tcs.add(in_edge)
        tc.append(in_edge)
    i += 1
  return tc


if __name__ == '__main__':
  t = TournamentInstance.raw(nx.tournament.random_tournament(5, 40))
  # plot and show graph
  print("Copeland winners", copeland_winners(t))
  print("Slater winner:", slater_winner(t))
  nx.draw_circular(t.graph, with_labels=True)
  plt.show()
