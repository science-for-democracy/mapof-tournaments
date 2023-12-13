import networkx as nx
import matplotlib.pyplot as plt

from mapel.tournaments.objects.GraphSimilarity import ged_blp, ged_blp_faster
from mapel.tournaments.objects.TournamentCultures import ordered
from mapel.tournaments.objects.TournamentInstance import TournamentInstance


def get_copeland_winners(tournament, num_winners=1):
  """
    Returns the Copeland winners of the given tournament.
    """
  copeland_scores = tournament.graph.out_degree()
  winners = list(sorted(copeland_scores, key=lambda x: x[1], reverse=True))[:num_winners]
  return winners


def get_slater_winner(tournament):
  """
    Returns one of the Slater winners of the given tournament.
    """
  transitive = ordered(tournament.graph.number_of_nodes())
  _distance, matching = ged_blp_faster(transitive, tournament.graph)
  # print("###slater: ", _distance, matching)
  winner = next(pair[1] for pair in matching if pair[0] == 0)
  return winner


if __name__ == '__main__':
  t = TournamentInstance.raw(nx.tournament.random_tournament(5, 40))
  # plot and show graph
  print("Copeland winners", get_copeland_winners(t, 2))
  print("Slater winner:", get_slater_winner(t))
  nx.draw_circular(t.graph, with_labels=True)
  plt.show()
