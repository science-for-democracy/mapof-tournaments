from collections import defaultdict
from random import uniform

import networkx as nx
import numpy as np
from mapel.core.objects.Instance import Instance
from mapel.tournaments.objects.TournamentCultures import (ordered,
                                                          rock_paper_scissors)


class TournamentInstance(Instance):

  def __init__(self,
               graph,
               experiment_id: str,
               instance_id: str,
               family_id: str,
               culture_id: str = 'none',
               alpha: float = np.NaN) -> None:
    super().__init__(experiment_id=experiment_id,
                     instance_id=instance_id,
                     culture_id=culture_id,
                     alpha=alpha)
    if isinstance(graph, list):
      graph = np.array(graph)
    if isinstance(graph, np.ndarray):
      self.graph = (nx.from_numpy_array(graph, create_using=nx.DiGraph()))
    else:
      self.graph = graph

    self.sp_matrix = None

  def to_shortest_paths_matrix(self):
    if self.sp_matrix is not None:
      return self.sp_matrix
    sp_matrix = nx.to_numpy_array(self.graph)
    n = len(sp_matrix)
    sp_matrix = np.where(sp_matrix == 0, n, sp_matrix)
    for i in range(n):
      sp_matrix[i, i] = 0
    # Floyd-Warshall
    for k in range(n):
      for i in range(n):
        for j in range(n):
          sp_matrix[i, j] = min(sp_matrix[i, j], sp_matrix[i, k] + sp_matrix[k, j])
    self.sp_matrix = sp_matrix
    return sp_matrix

  @staticmethod
  def raw(graph):
    return TournamentInstance(graph, 'none', 'none', 'none')

  @staticmethod
  def from_compass(num_participants: int,
                   compass: str,
                   experiment_id: str,
                   instance_id: str,
                   culture_id: str = 'none'):
    compass = compass.lower()
    if compass == 'ordered':
      adjacency_matrix = ordered(num_participants)
      return [
          TournamentInstance(adjacency_matrix, experiment_id, instance_id, culture_id)
      ]
    elif compass == 'rock-paper-scissors':
      adjacency_matrix = rock_paper_scissors(num_participants)
      return [
          TournamentInstance(adjacency_matrix, experiment_id, instance_id, culture_id)
      ]
    elif compass == 'o2rps':
      tournaments = []
      for i, n in enumerate(range(1, num_participants - 2)):
        g1 = TournamentInstance.from_compass(n,
                                             'ordered',
                                             experiment_id,
                                             instance_id,
                                             culture_id)[0].graph
        g2 = TournamentInstance.from_compass(num_participants - n,
                                             'rock-paper-scissors',
                                             experiment_id,
                                             instance_id,
                                             culture_id)[0].graph
        g2 = nx.relabel_nodes(g2, {i: i + n
                                   for i in g2.nodes})

        g = nx.compose(g1, g2)
        for i in g1.nodes:
          for j in g2.nodes:
            g.add_edge(i, j)
        tournaments.append(
            TournamentInstance(g, experiment_id, f"{instance_id}_{i}", culture_id))
      return tournaments
    elif compass == 'rps2o':
      tournaments = []
      for i, n in enumerate(range(1, num_participants - 2)):
        g1 = TournamentInstance.from_compass(n,
                                             'ordered',
                                             experiment_id,
                                             instance_id,
                                             culture_id)[0].graph
        g2 = TournamentInstance.from_compass(num_participants - n,
                                             'rock-paper-scissors',
                                             experiment_id,
                                             instance_id,
                                             culture_id)[0].graph
        g2 = nx.relabel_nodes(g2, {i: i + n
                                   for i in g2.nodes})
        g = nx.compose(g1, g2)
        for i in g1.nodes:
          for j in g2.nodes:
            g.add_edge(j, i)
        tournaments.append(
            TournamentInstance(g, experiment_id, f"{instance_id}_{i}", culture_id))
      return tournaments
    elif compass == 'test':

      def force_edge(g, i, j):
        if i == j:
          return
        if g.has_edge(j, i):
          g.remove_edge(j, i)
        if g.has_edge(i, j):
          return
        g.add_edge(i, j)

      graphs = []
      if num_participants % 3 != 0:
        graphs.append(
            TournamentInstance.from_compass(num_participants % 3,
                                            'rock-paper-scissors',
                                            experiment_id,
                                            instance_id,
                                            culture_id)[0].graph)
      for i in range(num_participants % 3, num_participants, 3):
        g = TournamentInstance.from_compass(3,
                                            'rock-paper-scissors',
                                            experiment_id,
                                            instance_id,
                                            culture_id)[0].graph
        g = nx.relabel_nodes(g, {j: j + i
                                 for j in g.nodes})
        graphs.append(g)
      g = graphs[0]
      for i in range(1, len(graphs)):
        g = nx.compose(g, graphs[i])
      for g1 in range(len(graphs)):
        for g2 in range(g1 + 1, len(graphs)):
          for i in graphs[g1].nodes:
            for j in graphs[g2].nodes:
              force_edge(g, i, j)

      return [TournamentInstance(g, experiment_id, instance_id, culture_id)]
    else:
      raise ValueError(
          f'Compass {compass} not supported. Supported compasses are: ordered, rock-paper-scissors, mixed.'
      )

  @staticmethod
  def from_weights(weights,
                   experiment_id: str,
                   instance_id: str,
                   culture_id: str = 'none'):
    n = len(weights)
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
      for j in range(i + 1, n):
        p1 = weights[i] / (weights[i] + weights[j])
        if uniform(0, 1) < p1:
          adjacency_matrix[i, j] = 1
        else:
          adjacency_matrix[j, i] = 1
    return TournamentInstance(adjacency_matrix, experiment_id, instance_id, culture_id)

  @staticmethod
  # TODO: Should this randomize weights on draws?
  def from_election(election):
    n = election.num_candidates
    adjacency_matrix = np.zeros((n, n))
    pairwise_matrix = election.votes_to_pairwise_matrix()
    for i in range(n):
      for j in range(i + 1, n):
        if pairwise_matrix[i, j] > pairwise_matrix[j, i]:
          adjacency_matrix[i, j] = 1
        else:
          adjacency_matrix[j, i] = 1
    return TournamentInstance(adjacency_matrix,
                              election.experiment_id,
                              election.instance_id,
                              election.culture_id)

  @staticmethod
  def from_bridge_df(df, experiment_id: str, instance_id: str, culture_id: str = 'none'):
    rr = df[df['Phase'] == 'RR']
    graph = nx.DiGraph()
    graph.add_nodes_from(rr['t1'])
    for _, row in rr.iterrows():
      if row['resultVPsHome'] > row['resultVPsVis']:
        graph.add_edge(row['t1'], row['t2'])
      elif row['resultVPsHome'] < row['resultVPsVis']:
        graph.add_edge(row['t2'], row['t1'])
      else:
        print('draw')
    return TournamentInstance(graph, experiment_id, instance_id, culture_id)

  @staticmethod
  def from_dict_of_lists(d,
                         experiment_id: str,
                         instance_id: str,
                         culture_id: str = 'none'):
    return TournamentInstance(nx.from_dict_of_lists(d, create_using=nx.DiGraph),
                              experiment_id,
                              instance_id,
                              culture_id)

  @staticmethod
  def from_nba_df(df,
                  params,
                  experiment_id: str,
                  instance_id: str,
                  culture_id: str = 'none'):
    # get all unique GAME_IDs
    game_ids = df['GAME_ID'].unique()
    # get all unique TEAM_ABBREVIATIONs
    team_abbrs = df['TEAM_ABBREVIATION'].unique()
    wins = {t: 0
            for t in team_abbrs}
    results = defaultdict(lambda: defaultdict(lambda: {
        'pm': 0, 'hw': 0, 'pf': 0
    }))
    for game_id in game_ids:
      gf = df[df['GAME_ID'] == game_id]
      t1, t2 = gf['TEAM_ABBREVIATION']
      pt1, pt2 = gf['PTS']
      pf1, pf2 = gf['PF']
      if pt1 > pt2:
        wins[t1] += 1
      else:
        wins[t2] += 1
      results[t1][t2]['pm'] += pt1
      results[t1][t2]['pm'] -= pt2
      results[t1][t2]['hw'] = max(results[t1][t2]['hw'], pt1 - pt2)
      results[t1][t2]['pf'] += pf1
      results[t2][t1]['pm'] += pt2
      results[t2][t1]['pm'] -= pt1
      results[t2][t1]['hw'] = max(results[t2][t1]['hw'], pt2 - pt1)
      results[t2][t1]['pf'] += pf2
    graph = nx.DiGraph()
    # Get top 20
    if 'top' in params:
      qualified_teams = sorted(wins, key=wins.get, reverse=True)[:params['top']]
    else:
      qualified_teams = team_abbrs
    graph.add_nodes_from(qualified_teams)
    n = len(qualified_teams)
    for i in range(n):
      for j in range(i + 1, n):
        t1, t2 = qualified_teams[i], qualified_teams[j]
        comp1 = results[t1][t2]
        comp2 = results[t2][t1]
        # Winner is decided by:
        # 1. Point difference
        # 2. Highest versus win
        # 3. Least amount of fouls
        comp1 = (comp1['pm'], comp1['hw'], comp1['pf'])
        comp2 = (comp2['pm'], comp2['hw'], comp2['pf'])
        if comp1 > comp2:
          graph.add_edge(t1, t2)
        elif comp2 > comp1:
          graph.add_edge(t2, t1)
        else:
          # TODO: make a better draw resolution
          graph.add_edge(t1, t2)
          print(instance_id, 'draw')
    return TournamentInstance(graph, experiment_id, instance_id, culture_id)

  @staticmethod
  def sample_tournament(base_instance,
                        num_participants: int,
                        experiment_id: str,
                        instance_id: str,
                        culture_id: str = 'none'):
    graph = base_instance.graph.copy()
    nodes = list(graph.nodes)
    graph.remove_nodes_from(
        np.random.choice(nodes, len(nodes) - num_participants, replace=False))
    return TournamentInstance(graph, experiment_id, instance_id, culture_id)

  # def save_graph_plot(self, path, **kwargs):
  #   fig = plt.figure()
  #   nx.draw_circular(self.graph,
  #                    ax=fig.add_subplot(),
  #                    labels=dict(self.graph.out_degree()),
  #                    **kwargs)
  #   plt.savefig(path)
  #   plt.close('all')
  def _to_graphviz(self, relabel=False, **kwargs):
    import pygraphviz as pgv
    d = {}
    for i in self.graph.nodes:
      d[i] = {}
    for i, j in self.graph.edges:
      d[i][j] = None
    G = pgv.AGraph(d, directed=True)
    G.node_attr["shape"] = "circle"
    G.edge_attr.update(len="2.0")
    G.layout(prog="circo")
    if relabel:
      for i in G.nodes():
        G.get_node(i).attr["label"] = G.out_degree(i)
    return G

  def save_graph_plot(self, path, relabel=False, **kwargs):
    G = self._to_graphviz(relabel=relabel)
    if path[:-4] != '.png':
      path = path + '.png'
    G.draw(path, **kwargs)
