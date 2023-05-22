import itertools
import json
import pickle
import os
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from random import uniform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import json_dump
from numpy.lib.twodim_base import triu_indices

import networkx as nx
from mapel.core.objects.Experiment import Experiment
from mapel.core.objects.Family import Family
from mapel.core.objects.Instance import Instance
from mapel.elections.objects.ElectionFamily import ElectionFamily
from mapel.tournaments.objects.GraphSimilarity import (Distances,
                                                       get_similarity_measure,
                                                       parallel_runner)
from progress.bar import Bar
from tqdm.contrib.concurrent import process_map


class TournamentInstance(Instance):

    def __init__(self,
                 graph,
                 experiment_id: str,
                 instance_id: str,
                 culture_id: str = 'none',
                 alpha: float = np.NaN) -> None:
        super().__init__(experiment_id=experiment_id,
                         instance_id=instance_id,
                         culture_id=culture_id,
                         alpha=alpha)
        if isinstance(graph, list):
            graph = np.array(graph)
        if isinstance(graph, np.ndarray):
            self.graph = (nx.from_numpy_array(graph,
                                              create_using=nx.DiGraph()))
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
                    sp_matrix[i, j] = min(sp_matrix[i, j],
                                          sp_matrix[i, k] + sp_matrix[k, j])
        self.sp_matrix = sp_matrix
        return sp_matrix

    @staticmethod
    def from_compass(num_participants: int,
                     compass: str,
                     experiment_id: str,
                     instance_id: str,
                     culture_id: str = 'none'):
        compass = compass.lower()
        if compass == 'ordered':
            adjacency_matrix = np.zeros((num_participants, num_participants))
            for i in range(num_participants):
                for j in range(i + 1, num_participants):
                    adjacency_matrix[i, j] = 1
            return TournamentInstance(adjacency_matrix, experiment_id,
                                      instance_id, culture_id)
        elif compass == 'rock-paper-scissors':
            adjacency_matrix = np.zeros((num_participants, num_participants))
            for jump_length in range(1, num_participants // 2 + 1):
                # For the last iteration with even number of participants, we only set half of the edges.
                for i in range(num_participants if jump_length <
                               (num_participants + 1) //
                               2 else num_participants // 2):
                    j = (i + jump_length) % num_participants
                    adjacency_matrix[i, j] = 1
            return TournamentInstance(adjacency_matrix, experiment_id,
                                      instance_id, culture_id)
        elif compass == 'mixed':
            g1 = TournamentInstance.from_compass((num_participants + 1) // 2,
                                                 'ordered', experiment_id,
                                                 instance_id, culture_id).graph
            g2 = TournamentInstance.from_compass(num_participants // 2,
                                                 'rock-paper-scissors',
                                                 experiment_id, instance_id,
                                                 culture_id).graph
            g2 = nx.relabel_nodes(
                g2, {i: i + (num_participants + 1) // 2
                     for i in g2.nodes})
            g = nx.compose(g1, g2)
            for i in g1.nodes:
                for j in g2.nodes:
                    g.add_edge(i, j)
            return TournamentInstance(g, experiment_id, instance_id,
                                      culture_id)
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
                                                    experiment_id, instance_id,
                                                    culture_id).graph)
            for i in range(num_participants % 3, num_participants, 3):
                g = TournamentInstance.from_compass(3, 'rock-paper-scissors',
                                                    experiment_id, instance_id,
                                                    culture_id).graph
                g = nx.relabel_nodes(g, {j: j + i for j in g.nodes})
                graphs.append(g)
            g = graphs[0]
            for i in range(1, len(graphs)):
                g = nx.compose(g, graphs[i])
            for g1 in range(len(graphs)):
                for g2 in range(g1 + 1, len(graphs)):
                    for i in graphs[g1].nodes:
                        for j in graphs[g2].nodes:
                            force_edge(g, i, j)

            return TournamentInstance(g, experiment_id, instance_id,
                                      culture_id)
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
        return TournamentInstance(adjacency_matrix, experiment_id, instance_id,
                                  culture_id)

    @staticmethod
    def from_election(election):
        n = election.num_candidates
        adjacency_matrix = np.zeros((n, n))
        pairwise_matrix = election.votes_to_pairwise_matrix()
        print(pairwise_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if pairwise_matrix[i, j] > pairwise_matrix[j, i]:
                    adjacency_matrix[i, j] = 1
                else:
                    adjacency_matrix[j, i] = 1
        return TournamentInstance(adjacency_matrix, election.experiment_id,
                                  election.instance_id, election.culture_id,
                                  election.alpha)

    def save_graph_plot(self, path, **kwargs):
        fig = plt.figure()
        nx.draw_circular(self.graph,
                         ax=fig.add_subplot(),
                         labels=dict(self.graph.out_degree()),
                         **kwargs)
        plt.savefig(path)
        plt.close('all')


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

    def prepare_tournament_family(self):
        if self.single:
            if 'compass' in self.params:
                return {
                    self.family_id:
                    TournamentInstance.from_compass(self.num_participants,
                                                    self.params['compass'],
                                                    self.experiment_id,
                                                    self.family_id,
                                                    self.culture_id)
                }

            elif 'adjacency_matrix' in self.params:
                adjacency_matrix = self.params['adjacency_matrix']
                return {
                    self.family_id:
                    TournamentInstance(adjacency_matrix, self.experiment_id,
                                       self.family_id, self.culture_id)
                }
            else:
                raise ValueError(
                    'Either compass or adjacency_matrix must be specified for a single tournament.'
                )
        else:
            weights = self.params['weights']
            self.num_participants = len(weights)
            tournaments = {}
            for i in range(self.size):
                instance_id = f'{self.family_id}_{i}'
                tournaments[instance_id] = TournamentInstance.from_weights(
                    weights, self.experiment_id, instance_id, self.culture_id)
            return tournaments

    def prepare_from_ordinal_election_family(self):
        num_voters = self.params[
            'num_voters'] if 'num_voters' in self.params else self.num_participants
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
        return tournaments

    def prepare_family(self, plot_path=None):
        if self.instance_type == 'tournament':
            family = self.prepare_tournament_family()
        elif self.instance_type == 'ordinal':
            family = self.prepare_from_ordinal_election_family()
        else:
            raise ValueError(
                f'Instance type {self.instance_type} not supported.')
        if plot_path is not None:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            for k, v in family.items():
                v.save_graph_plot(os.path.join(plot_path, str(k)))
        return family


class TournamentExperiment(Experiment):

    def __init__(self,
                 instances=None,
                 distances=None,
                 dim=2,
                 store=True,
                 coordinates=None,
                 distance_id='emd-positionwise',
                 experiment_id=None,
                 _import=True,
                 clean=False,
                 coordinates_names=None,
                 embedding_id='kamada',
                 fast_import=False,
                 with_matrix=False):
        super().__init__(instances=instances,
                         distances=distances,
                         dim=dim,
                         store=store,
                         coordinates=coordinates,
                         distance_id=distance_id,
                         experiment_id=experiment_id,
                         _import=_import,
                         clean=clean,
                         coordinates_names=coordinates_names,
                         embedding_id=embedding_id,
                         fast_import=fast_import,
                         with_matrix=with_matrix)
        self.instances = {}
        self.distances = defaultdict(dict)
        self.families = None

    # def add_tournament(self, graph, instance_id):
    #     self.instances[instance_id] = TournamentInstance(
    #         graph, culture_id=instance_id)

    def add_family(self,
                   culture_id: str = "none",
                   params: dict = dict(),
                   size: int = 1,
                   label: str = 'none',
                   color: str = "black",
                   alpha: float = 1.,
                   show: bool = True,
                   marker: str = 'o',
                   starting_from: int = 0,
                   num_participants: int = 10,
                   family_id: str | None = None,
                   single: bool = False,
                   path: dict = dict(),
                   plot_path=None,
                   instance_type='tournament',
                   tournament_id: str | None = None):
        if tournament_id is not None:
            family_id = tournament_id

        if self.families is None:
            self.families = {}

        if family_id is None:
            family_id = culture_id + '_' + str(num_participants)
            if culture_id in {'urn_model'
                              } and params and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif culture_id in {'mallows'
                                } and params and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif culture_id in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

        if label == 'none':
            label = family_id

        self.families[family_id] = TournamentFamily(
            culture_id=culture_id,
            family_id=family_id,
            params=params,
            label=label,
            color=color,
            alpha=alpha,
            show=show,
            size=size,
            marker=marker,
            starting_from=starting_from,
            num_participants=num_participants,
            path=path,
            single=single,
            instance_type=instance_type)

        new_instances = self.families[family_id].prepare_family(
            plot_path=plot_path)

        for instance_id in new_instances:
            self.instances[instance_id] = new_instances[instance_id]

        self.families[family_id].instance_ids = list(new_instances.keys())

        return list(new_instances.keys())

    def _compute_distances(self, metric):
        n = len(self.instances)
        bar = Bar('Computing distances:', max=n * (n - 1) // 2)
        bar.start()
        for e, (i, t1) in enumerate(self.instances.items()):
            for j, t2 in list(self.instances.items())[e + 1:]:
                self.distances[j][i] = self.distances[i][j] = metric(t1, t2)
                bar.next()

    def _compute_distances_parallel(self, metric):
        n = len(self.instances)
        instance_ids = list(self.instances.keys())
        tournaments = list(self.instances.values())
        indices = list(zip(*triu_indices(n, 1)))
        work = [(metric, tournaments[i], tournaments[j]) for i, j in indices]
        with Pool() as p:
            distances = list(
                process_map(parallel_runner, work, total=len(work)))
            # distances = p.starmap(metric, work)
        for d, (i, j) in zip(distances, indices):
            self.distances[instance_ids[j]][instance_ids[i]] = self.distances[
                instance_ids[i]][instance_ids[j]] = d

    def compute_distances(self,
                          metric: Distances = Distances.GED_OPT,
                          parallel: bool = False,
                          path: str = '',
                          **kwargs):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                self.distances = pickle.load(f)
        else:
            if parallel:
                self._compute_distances_parallel(
                    get_similarity_measure(metric, **kwargs))
            else:
                self._compute_distances(
                    get_similarity_measure(metric, **kwargs))
            if path:
                with open(path, 'wb') as f:
                    pickle.dump(self.distances, f)

        if isinstance(self.distances, dict):
            print(json.dumps(self.distances, indent=4))
        else:
            print(self.distances)
        # Print top 10 largest distances
        all = []
        for k, v in self.distances.items():
            for k2, v2 in v.items():
                all.append((k, k2, v2))
        top = sorted(all, key=lambda x: x[2], reverse=True)[:250]
        print(''.join([str(x) + '\n' for x in top]))

    def save_tournament_plots(self, path: str = 'graphs'):
        if not os.path.exists(path):
            os.makedirs(path)
        for k, v in self.instances.items():
            v.save_graph_plot(os.path.join(path, str(k)))
