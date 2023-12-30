import ast
import csv
import itertools
import json
import os
import pickle
import time
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from random import uniform

import mapel.core.persistence.experiment_exports as exports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pandas_access as mdb
from mapel.core.objects.Experiment import Experiment
from mapel.core.objects.Family import Family
from mapel.core.objects.Instance import Instance
from mapel.core.utils import make_folder_if_do_not_exist
from mapel.elections.objects.ElectionFamily import ElectionFamily
from mapel.tournaments.objects.Features import get_feature
from mapel.tournaments.objects.TournamentFamily import TournamentFamily
from mapel.tournaments.objects.TournamentSimilarity import (get_distance,
                                                            parallel_runner)
from matplotlib.font_manager import json_dump
from numpy.lib.twodim_base import triu_indices
from progress.bar import Bar
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class TournamentExperiment(Experiment):

  def __init__(self,
               instances=None,
               distances=None,
               coordinates=None,
               distance_id="ged_blp",
               experiment_id=None,
               coordinates_names=None,
               embedding_id='mds',
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
    self.instance_type = 'tournaments'

  def __str__(self):
    return f"TournamentExperiment: {self.experiment_id}\n {len(self.instances) if self.instances else None} instances\n {len(self.distances) if self.distances else None} distances\n {len(self.coordinates) if self.coordinates else None} coordinates"

  def create_structure(self) -> None:
    if not os.path.isdir("experiments/"):
      os.mkdir(os.path.join(os.getcwd(), "experiments"))

    if not os.path.isdir("images/"):
      os.mkdir(os.path.join(os.getcwd(), "images"))
    if not os.path.isdir("jsons/"):
      os.mkdir(os.path.join(os.getcwd(), "jsons"))

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
      os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "tournaments"))
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
    if not os.path.exists(path):
      print("File not found. Creating new experiment...")
      self.create_structure()
      exit(0)
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

        # Ordinal-compatibility
        if 'num_candidates' in row.keys():
          num_participants = int(row['num_candidates'])

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
          show = row['show'].strip() == 'True'

        if 'label' in row.keys():
          label = str(row['label'])

        # if 'path' in row.keys():
        #     path = row['path'].strip()

        if 'params' in row.keys():
          params = ast.literal_eval(str(row['params']))

        if 'num_voters' in row.keys():
          params['num_voters'] = int(row['num_voters'])

        single = size == 1 and 'compass' not in params

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
    instance_ids = list(self.instances.keys())
    indices = list(zip(*triu_indices(n, 1)))
    indices_to_compute = [
        (i, j) for i,
        j in indices
        if instance_ids[i] not in self.distances or instance_ids[j] not in self.distances
    ]
    bar = Bar('Computing distances:', max=len(indices_to_compute))
    bar.start()
    for i, j in indices_to_compute:
      self.distances.setdefault(instance_ids[i], dict())
      self.distances.setdefault(instance_ids[j], dict())
      self.distances[instance_ids[j]][instance_ids[i]] = self.distances[instance_ids[i]][
          instance_ids[j]] = metric(self.instances[instance_ids[i]],
                                    self.instances[instance_ids[j]])
      bar.next()

  def _compute_distances_parallel(self, metric):
    n = len(self.instances)
    instance_ids = list(self.instances.keys())
    tournaments = list(self.instances.values())
    indices = list(zip(*triu_indices(n, 1)))
    indices_to_compute = [
        (i, j) for i,
        j in indices
        if instance_ids[i] not in self.distances or instance_ids[j] not in self.distances
    ]
    work = [(metric, tournaments[i], tournaments[j]) for i, j in indices_to_compute]
    # with Pool() as p:
    #   distances = list(
    #       process_map(parallel_runner,
    #                   work,
    #                   total=len(work),
    #                   chunksize=max(1, len(work) // (5000 * os.cpu_count()))))
    with Pool() as p:
      distances = list(tqdm(p.imap(parallel_runner, work), total=len(work)))
    for d, (i, j) in zip(distances, indices_to_compute):
      self.distances.setdefault(instance_ids[i], dict())
      self.distances.setdefault(instance_ids[j], dict())
      self.distances[instance_ids[j]][instance_ids[i]] = self.distances[instance_ids[i]][
          instance_ids[j]] = d

  def _store_distances_to_file(self, distance_id, distances, times, self_distances):
    print(os.getcwd())
    print(self.experiment_id)
    path_to_folder = os.path.join(os.getcwd(),
                                  "experiments",
                                  self.experiment_id,
                                  "distances")
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

  def compute_distances(self,
                        distance_id: str = "ged_blp",
                        parallel: bool = False,
                        clean: bool = False,
                        print_top=False):
    if not self.distances or clean:
      print(f"Generating {distance_id} from scratch...")
      self.distances = dict()
    if distance_id:
      self.distance_id = distance_id
    if parallel:
      self._compute_distances_parallel(get_distance(distance_id))
    else:
      self._compute_distances(get_distance(distance_id))

    if print_top:
      if isinstance(self.distances, dict):
        print(json.dumps(self.distances, indent=4))
      else:
        print(self.distances)
      all = []
      for k, v in self.distances.items():
        for k2, v2 in v.items():
          all.append((k, k2, v2))
      top = sorted(all, key=lambda x: x[2], reverse=True)[:250]
      print(''.join([str(x) + '\n' for x in top]))
    self._store_distances_to_file(self.distance_id, self.distances, None, False)

  def save_tournament_plots(self, path: str = 'graphs', **kwargs):
    if not os.path.exists(path):
      os.makedirs(path)
    for k, v in self.instances.items():
      v.save_graph_plot(os.path.join(path, str(k), **kwargs))

  def _compute_feature(self, feature_fun):
    feature_dict = {
        'value': {}, 'time': {}
    }
    for instance_id in tqdm(self.instances,
                            desc=f'Computing feature: {feature_fun.__name__}'):
      instance = self.instances[instance_id]
      start = time.time()
      feature_dict['value'][instance_id] = feature_fun(instance)
      feature_dict['time'][instance_id] = time.time() - start
    return feature_dict

  def _compute_feature_parallel(self, feature_fun):
    feature_dict = {
        'value': {}, 'time': {}
    }
    work = [(feature_fun, self.instances[instance_id]) for instance_id in self.instances]
    with Pool() as p:
      values = list(
          tqdm(p.imap(parallel_runner, work),
               total=len(work),
               desc=f'Computing feature: {feature_fun.__name__}'))
    for instance_id, value in zip(work, values):
      feature_dict['value'][instance_id] = feature_fun(instance_id)
      feature_dict['time'][instance_id] = -1
    return feature_dict

  def compute_feature(self,
                      feature_id,
                      feature_long_id=None,
                      saveas=None,
                      clean=False,
                      **kwargs):
    """ Compute a feature for all instances in the experiment """
    feature_long_id = feature_id if feature_long_id is None else feature_long_id

    folder_path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "features")
    make_folder_if_do_not_exist(folder_path)
    saveas = feature_long_id if saveas is None else saveas
    filepath = os.path.join(folder_path, f'{saveas}.csv')
    if os.path.exists(filepath) and not clean:
      print(f"Feature {feature_id} already exists. Not calculating...")
      return
    feature_fun = get_feature(feature_id)

    if feature_id[-9:] == '_parallel':
      feature_dict = self._compute_feature_parallel(feature_fun)
    else:
      feature_dict = self._compute_feature(feature_fun)

    if self.is_exported:
      exports.export_feature_to_file(self, feature_id, saveas, feature_dict)

    self.features[saveas] = feature_dict
    return feature_dict
