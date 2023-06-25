import ast
import csv
import itertools
import json
import pickle
import os
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.font_manager import json_dump
from numpy.lib.twodim_base import triu_indices
from progress.bar import Bar
from tqdm.contrib.concurrent import process_map

from mapel.core.objects.Experiment import Experiment
from mapel.core.objects.Family import Family
from mapel.core.objects.Instance import Instance
from mapel.core.utils import make_folder_if_do_not_exist
from mapel.elections.objects.ElectionFamily import ElectionFamily
from mapel.tournaments.objects.GraphSimilarity import (Distances, get_similarity_measure, parallel_runner,
                                                       ged_blp)


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
            return TournamentInstance(adjacency_matrix, experiment_id, instance_id, culture_id)
        elif compass == 'rock-paper-scissors':
            adjacency_matrix = np.zeros((num_participants, num_participants))
            for jump_length in range(1, num_participants // 2 + 1):
                # For the last iteration with even number of participants, we only set half of the edges.
                for i in range(num_participants if jump_length < (num_participants + 1) //
                               2 else num_participants // 2):
                    j = (i + jump_length) % num_participants
                    adjacency_matrix[i, j] = 1
            return TournamentInstance(adjacency_matrix, experiment_id, instance_id, culture_id)
        elif compass == 'mixed':
            g1 = TournamentInstance.from_compass((num_participants + 1) // 2, 'ordered', experiment_id,
                                                 instance_id, culture_id).graph
            g2 = TournamentInstance.from_compass(num_participants // 2, 'rock-paper-scissors', experiment_id,
                                                 instance_id, culture_id).graph
            g2 = nx.relabel_nodes(g2, {i: i + (num_participants + 1) // 2 for i in g2.nodes})
            g = nx.compose(g1, g2)
            for i in g1.nodes:
                for j in g2.nodes:
                    g.add_edge(i, j)
            return TournamentInstance(g, experiment_id, instance_id, culture_id)
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
                    TournamentInstance.from_compass(num_participants % 3, 'rock-paper-scissors', experiment_id,
                                                    instance_id, culture_id).graph)
            for i in range(num_participants % 3, num_participants, 3):
                g = TournamentInstance.from_compass(3, 'rock-paper-scissors', experiment_id, instance_id,
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

            return TournamentInstance(g, experiment_id, instance_id, culture_id)
        else:
            raise ValueError(
                f'Compass {compass} not supported. Supported compasses are: ordered, rock-paper-scissors, mixed.'
            )

    @staticmethod
    def from_weights(weights, experiment_id: str, instance_id: str, culture_id: str = 'none'):
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
        return TournamentInstance(adjacency_matrix, election.experiment_id, election.instance_id,
                                  election.culture_id, election.alpha)

    def save_graph_plot(self, path, **kwargs):
        fig = plt.figure()
        nx.draw_circular(self.graph, ax=fig.add_subplot(), labels=dict(self.graph.out_degree()), **kwargs)
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

    NON_ISO_FILENAME_PREF = 'all-non-isomorphic-'

    def get_all_non_isomorphic_graphs(self, n):
        """Generate all non-isomorphic graphs with n nodes."""
        pickle_path = (os.path.join(os.getcwd(), "experiments", self.experiment_id, "pickles"))
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle_file = os.path.join(pickle_path, f'{TournamentFamily.NON_ISO_FILENAME_PREF}{n}.pkl')
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
        for g1 in gen(np.zeros((n, n)), 0):
            g1 = nx.from_numpy_array(g1, create_using=nx.DiGraph())
            for g2 in graphs:
                ged = ged_blp(g1, g2) < 0.5
                iso = nx.is_isomorphic(g1, g2)
                if ged and iso:
                    break
                if ged != iso:
                    raise Exception('GED and isomorphism are not consistent')
            else:
                graphs.append(g1)
                print(len(graphs))
                # if (len(graphs) == COUNT[n]):
                #     break
        pickle.dump(graphs, open(pickle_file, 'wb'))
        return graphs

    def prepare_tournament_family(self):
        if self.single:
            if 'compass' in self.params:
                return {
                    self.family_id:
                    TournamentInstance.from_compass(self.num_participants, self.params['compass'],
                                                    self.experiment_id, self.family_id, self.culture_id)
                }

            elif 'adjacency_matrix' in self.params:
                adjacency_matrix = self.params['adjacency_matrix']
                return {
                    self.family_id:
                    TournamentInstance(adjacency_matrix, self.experiment_id, self.family_id, self.culture_id)
                }
            else:
                raise ValueError('Either compass or adjacency_matrix must be specified for a single tournament.')
        elif self.culture_id.startswith('all-non-isomorphic'):
            n = int(self.culture_id.split('-')[-1])
            tournaments = {}
            for i, g in enumerate(self.get_all_non_isomorphic_graphs(n)):
                instance_id = f'{self.family_id}_{i}'
                tournaments[instance_id] = TournamentInstance(g, self.experiment_id, instance_id,
                                                              self.culture_id)
            return tournaments
        else:
            weights = self.params['weights']
            self.num_participants = len(weights)
            tournaments = {}
            for i in range(self.size):
                instance_id = f'{self.family_id}_{i}'
                tournaments[instance_id] = TournamentInstance.from_weights(weights, self.experiment_id,
                                                                           instance_id, self.culture_id)
            return tournaments

    def prepare_from_ordinal_election_family(self):
        num_voters = self.params['num_voters'] if 'num_voters' in self.params else self.num_participants
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
        return tournaments


class TournamentExperiment(Experiment):

    def __init__(self,
                 instances=None,
                 distances=None,
                 coordinates=None,
                 distance_id=Distances.GED_BLP,
                 experiment_id=None,
                 coordinates_names=None,
                 embedding_id='kamada',
                 fast_import=False,
                 with_matrix=False):
        super().__init__(instances=instances,
                         distances=distances,
                         coordinates=coordinates,
                         distance_id=distance_id,
                         experiment_id=experiment_id,
                         coordinates_names=coordinates_names,
                         embedding_id=embedding_id,
                         fast_import=fast_import,
                         with_matrix=with_matrix)
        self.instances = {}
        self.distances = defaultdict(dict)
        self.families = {}

    def create_structure(self) -> None:
        if not os.path.isdir("experiments/"):
            os.mkdir(os.path.join(os.getcwd(), "experiments"))

        if not os.path.isdir("images/"):
            os.mkdir(os.path.join(os.getcwd(), "images"))

        if not os.path.isdir("trash/"):
            os.mkdir(os.path.join(os.getcwd(), "trash"))
        try:
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances"))
            # os.mkdir(
            #     os.path.join(os.getcwd(), "experiments", self.experiment_id,
            #                  "features"))
            # os.mkdir(
            #     os.path.join(os.getcwd(), "experiments", self.experiment_id,
            #                  "coordinates"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "tournaments"))
            # os.mkdir(
            #     os.path.join(os.getcwd(), "experiments", self.experiment_id,
            #                  "matrices"))

            # PREPARE MAP.CSV FILE

            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")
            with open(path, 'w') as f:
                f.write(
                    "size;num_participants;culture_id;family_id;instance_type;color;marker;alpha;show;label;path;params\n"
                )
                print("Initialized empty experiment. Add families at " + path)
                exit(0)
        except FileExistsError:
            print("Experiment already exists!")

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        with open(path, 'r') as file_:

            header = [h.strip() for h in file_.readline().split(';')]
            reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

            # all_num_participants = []

            starting_from = 0
            for row in reader:

                size = 0
                num_participants = 0
                culture_id = 'none'
                family_id = 'none'
                instance_type = 'tournament'
                color = 'black'
                marker = 'o'
                alpha = 1.
                show = True
                label = 'none'
                params = dict()

                # try:
                #     if 'culture_id' in row.keys():
                #         culture_id = str(row['culture_id']).strip()
                # except:
                #     if 'model_id' in row.keys():
                #         culture_id = str(row['model_id']).strip()
                #     if 'culture_id' in row.keys():
                #         culture_id = str(row['culture_id']).strip()
                if 'size' in row.keys():
                    size = int(row['size'])

                if 'num_participants' in row.keys():
                    num_participants = int(row['num_participants'])

                if 'culture_id' in row.keys():
                    culture_id = str(row['culture_id']).strip()

                if 'family_id' in row.keys():
                    family_id = str(row['family_id'])

                if 'instance_type' in row.keys():
                    instance_type = str(row['instance_type']).strip()

                if 'color' in row.keys():
                    color = str(row['color']).strip()

                if 'marker' in row.keys():
                    marker = str(row['marker']).strip()

                if 'alpha' in row.keys():
                    alpha = float(row['alpha'])

                if 'show' in row.keys():
                    show = row['show'].strip() == 't'

                if 'label' in row.keys():
                    label = str(row['label'])

                # if 'path' in row.keys():
                #     path = ast.literal_eval(str(row['path']))

                if 'params' in row.keys():
                    params = ast.literal_eval(str(row['params']))

                single = size == 1

                families[family_id] = TournamentFamily(
                    experiment_id=self.experiment_id,
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
                    # path=path,
                    single=single,
                    instance_type=instance_type)
                starting_from += size

                # all_num_candidates.append(num_candidates)
                # all_num_voters.append(num_voters)

            # check_if_all_equal(all_num_candidates, 'num_candidates')
            # check_if_all_equal(all_num_voters, 'num_voters')

            self.num_families = len(families)
            self.num_elections = sum([families[family_id].size for family_id in families])
            self.main_order = [i for i in range(self.num_elections)]

        return families

    def add_instances_to_experiment(self):
        instances = {}

        for family_id in self.families:
            for instance in self.families[family_id].prepare_family().values():
                instances[instance.instance_id] = instance
        return instances

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
            if culture_id in {'urn_model'} and params and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif culture_id in {'mallows'} and params and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif culture_id in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

        if label == 'none':
            label = family_id

        self.families[family_id] = TournamentFamily(culture_id=culture_id,
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

        new_instances = self.families[family_id].prepare_family(plot_path=plot_path)

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
            distances = list(process_map(parallel_runner, work, total=len(work)))
            # distances = p.starmap(metric, work)
        for d, (i, j) in zip(distances, indices):
            self.distances[instance_ids[j]][instance_ids[i]] = self.distances[instance_ids[i]][
                instance_ids[j]] = d

    def _store_distances_to_file(self, distance_id, distances, times, self_distances):
        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder, f'{distance_id}.csv')

        with open(path_to_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])

            for i, election_1 in enumerate(self.distances.keys()):
                for j, election_2 in enumerate(self.distances.keys()):
                    if i < j or (i == j and self_distances):
                        distance = str(distances[election_1][election_2])
                        time_ = str(times[election_1][election_2]) if times else 0
                        writer.writerow([election_1, election_2, distance, time_])

    def compute_distances(self, metric: Distances = Distances.GED_OPT, parallel: bool = False, **kwargs):
        if metric:
            self.distance_id = metric
        if self.store:
            try:
                self.distances, _times, _stds, _mappings = self.add_distances_to_experiment()
                print("Distances loaded from file")
                print(self.distances)
            except FileNotFoundError:
                # load from pickle
                self.distances = pickle.load(
                    open(os.path.join(os.getcwd(), "distances", "full-non-isomorphic_7.pickle"), 'rb'))
                self._store_distances_to_file(metric, self.distances, None, False)
                print("Distances not found, computing them...")
                pass
        if len(self.distances) == 0:
            if parallel:
                self._compute_distances_parallel(get_similarity_measure(metric, **kwargs))
            else:
                self._compute_distances(get_similarity_measure(metric, **kwargs))
            if self.store:
                self._store_distances_to_file(metric, self.distances, None, False)

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
