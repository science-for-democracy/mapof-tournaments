import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mapel.tournaments.objects.helpers import \
    fill_with_losers_up_to_a_power_of_two
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


def single_elimination_can_player_win1(tournament, player, multithreaded=False):
  """Checks if a given player can win any single elimination tournament."""
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  g = fill_with_losers_up_to_a_power_of_two(tournament.graph)
  g = nx.convert_node_labels_to_integers(g, label_attribute='old_label')
  for node in g.nodes():
    if g.nodes[node]['old_label'] == player:
      player = node
      break
  n = len(g)
  levels = round(np.log2(n)) + 1
  m = gp.Model('single_elimination_can_player_win', env=env)
  if not multithreaded:
    m.setParam(gp.GRB.Param.Threads, 1)

  # Vertex to tree matching variables
  t = np.ndarray(shape=(n, 2 * n - 1), dtype=object)
  for vertex_ind in range(n):
    for tree_ind in range(2 * n - 1):
      t[vertex_ind, tree_ind] = m.addVar(vtype=gp.GRB.BINARY,
                                         name='t_{},{}'.format(vertex_ind, tree_ind))

  # Objective
  obj = t[player, 0]
  m.setObjective(obj, gp.GRB.MAXIMIZE)

  def level_start(level):
    return (1 << level) - 1

  def level_end(level):
    return (1 << (level + 1)) - 1

  def left(tree_ind):
    return 2 * tree_ind + 1

  def right(tree_ind):
    return 2 * tree_ind + 2

  # Force permutation of participants at leaf level
  start = level_start(levels - 1)
  end = level_end(levels - 1)
  for vertex_ind in range(n):
    m.addConstr(
        gp.quicksum(t[vertex_ind, tree_ind] for tree_ind in range(start, end)) == 1)

  # One participant per tree node
  for tree_ind in range(2 * n - 1):
    m.addConstr(gp.quicksum(t[vertex_ind, tree_ind] for vertex_ind in range(n)) == 1)

  # Winner is also one of the children
  for vertex_ind in range(n):
    for tree_ind in range(level_start(levels - 1)):
      m.addConstr(t[vertex_ind, tree_ind] <= t[vertex_ind, left(tree_ind)] +
                  t[vertex_ind, right(tree_ind)])

  # Winner is actually winning in g as well
  for vertex_ind in range(n):
    for tree_ind in range(level_start(levels - 1)):
      m.addConstr(t[vertex_ind, tree_ind] <= gp.quicksum([
          t[succ, left(tree_ind)] + t[succ, right(tree_ind)]
          for succ in g.successors(vertex_ind)
      ]))

  # Solve
  m.optimize()

  if m.Status != gp.GRB.OPTIMAL:
    raise GurobiException(f"Optimization failed... Status: {m.Status}", m.Status)

  matching = []
  for vertex_ind in range(n):
    for tree_ind in range(2 * n - 1):
      if t[vertex_ind, tree_ind].x > 0.5:
        matching.append((g.nodes[vertex_ind]['old_label'], tree_ind))
  return m.objVal, matching


def single_elimination_can_player_win2(tournament, player, multithreaded=False):
  """Checks if a given player can win any single elimination tournament.
  This one forces winners to always be the left child of the parent."""
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  g = fill_with_losers_up_to_a_power_of_two(tournament.graph)
  g = nx.convert_node_labels_to_integers(g, label_attribute='old_label')
  for node in g.nodes():
    if g.nodes[node]['old_label'] == player:
      player = node
      break
  n = len(g)
  levels = round(np.log2(n)) + 1
  m = gp.Model('single_elimination_can_player_win', env=env)
  if not multithreaded:
    m.setParam(gp.GRB.Param.Threads, 1)

  # Vertex to tree matching variables
  t = np.ndarray(shape=(n, 2 * n - 1), dtype=object)
  for vertex_ind in range(n):
    for tree_ind in range(2 * n - 1):
      t[vertex_ind, tree_ind] = m.addVar(vtype=gp.GRB.BINARY,
                                         name='t_{},{}'.format(vertex_ind, tree_ind))

  # Objective
  obj = t[player, 0]
  m.setObjective(obj, gp.GRB.MAXIMIZE)

  def level_start(level):
    return (1 << level) - 1

  def level_end(level):
    return (1 << (level + 1)) - 1

  def left(tree_ind):
    return 2 * tree_ind + 1

  def right(tree_ind):
    return 2 * tree_ind + 2

  # Force permutation of participants at leaf level
  start = level_start(levels - 1)
  end = level_end(levels - 1)
  for vertex_ind in range(n):
    m.addConstr(
        gp.quicksum(t[vertex_ind, tree_ind] for tree_ind in range(start, end)) == 1)

  # One participant per tree node
  for tree_ind in range(2 * n - 1):
    m.addConstr(gp.quicksum(t[vertex_ind, tree_ind] for vertex_ind in range(n)) == 1)

  # Winner is always on the left
  for tree_ind in range(level_end(levels - 2)):
    for vertex_ind in range(n):
      m.addConstr(t[vertex_ind, tree_ind] == t[vertex_ind, left(tree_ind)])

  # Left wins with right
  for tree_ind in range(1, level_end(levels - 1), 2):
    ti1 = tree_ind
    ti2 = tree_ind + 1
    for vertex_ind in range(n):
      m.addConstr(
          t[vertex_ind,
            ti1] <= gp.quicksum([t[succ, ti2] for succ in g.successors(vertex_ind)]))

  # Solve
  m.optimize()

  if m.Status == gp.GRB.INFEASIBLE:
    m.computeIIS()
    m.write("/tmp/model.ilp")
    raise GurobiException(f"Optimization failed... Model is infeasible", m.Status)

  if m.Status != gp.GRB.OPTIMAL:
    raise GurobiException(f"Optimization failed... Status: {m.Status}", m.Status)

  matching = []
  for vertex_ind in range(n):
    for tree_ind in range(2 * n - 1):
      if t[vertex_ind, tree_ind].x > 0.5:
        matching.append((g.nodes[vertex_ind]['old_label'], tree_ind))
  return m.objVal, matching


def single_elimination_can_player_win3(tournament, player, multithreaded=False):
  """Checks if a given player can win any single elimination tournament.
  This one forces winners to always be the left child of the parent."""
  env = gp.Env(empty=True)
  env.setParam('OutputFlag', 0)
  env.start()

  g = fill_with_losers_up_to_a_power_of_two(tournament.graph)
  g = nx.convert_node_labels_to_integers(g, label_attribute='old_label')
  for node in g.nodes():
    if g.nodes[node]['old_label'] == player:
      player = node
      break
  n = len(g)
  m = gp.Model('single_elimination_can_player_win', env=env)
  # m.setParam(gp.GRB.Param.TimeLimit, 2)
  if not multithreaded:
    m.setParam(gp.GRB.Param.Threads, 1)

  # Vertex to permutation matching variables
  t = np.ndarray(shape=(n, n), dtype=object)
  for vertex_ind in range(n):
    for pm_ind in range(n):
      t[vertex_ind, pm_ind] = m.addVar(vtype=gp.GRB.BINARY,
                                       name='t_{},{}'.format(vertex_ind, pm_ind))

  # Objective
  obj = t[player, 0]
  m.setObjective(obj, gp.GRB.MAXIMIZE)

  # Force permutation of participants at leaf level
  for vertex_ind in range(n):
    m.addConstr(gp.quicksum(t[vertex_ind, pm_ind] for pm_ind in range(n)) == 1)
  for pm_ind in range(n):
    m.addConstr(gp.quicksum(t[vertex_ind, pm_ind] for vertex_ind in range(n)) == 1)

  # Force such permutation that the winner is always the first element
  jump = 1
  while jump < n:
    for vertex_ind in range(n):
      for pm_ind in range(0, n, jump * 2):
        m.addConstr(t[vertex_ind, pm_ind] <= gp.quicksum(
            [t[succ, pm_ind + jump] for succ in g.successors(vertex_ind)]))
    jump *= 2

  # Solve
  m.optimize()

  if m.Status == gp.GRB.INFEASIBLE:
    m.computeIIS()
    m.write("/tmp/model.ilp")
    raise GurobiException(f"Optimization failed... Model is infeasible", m.Status)

  if m.Status != gp.GRB.OPTIMAL:
    raise GurobiException(f"Optimization failed... Status: {m.Status}", m.Status)

  matching = []
  for vertex_ind in range(n):
    for pm_ind in range(n):
      if t[vertex_ind, pm_ind].x > 0.5:
        matching.append((g.nodes[vertex_ind]['old_label'], pm_ind))
  return m.objVal, matching


single_elimination_can_player_win = single_elimination_can_player_win3

if __name__ == '__main__':
  t = TournamentInstance.raw(nx.tournament.random_tournament(5, 40))
  # plot and show graph
  # print("Copeland winners", copeland_winners(t))
  # print("Slater winner:", slater_winner(t))
  # nx.draw_circular(t.graph, with_labels=True)
  # plt.show()
  n = 20
  g = ordered(n, 1, {})[0]
  g = TournamentInstance.raw(g)
  for i in range(n):
    print(i, single_elimination_can_player_win1(g, i, multithreaded=False))
    # print(i, single_elimination_can_player_win(g, i, multithreaded=False))
